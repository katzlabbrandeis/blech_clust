from PIL import Image  # noqa
from io import BytesIO  # noqa
import os
import boto3
from datetime import datetime
# S3 configuration
S3_BUCKET = os.getenv('BLECH_S3_BUCKET', 'blech-pipeline-outputs')
import blech_clust.utils.blech_utils as bu  # noqa


def compress_image(image_path, max_size_kb=50):
    """Compress image to a maximum size in KB.

    Args:
        image_path (str): Path to the image file
        max_size_kb (int): Maximum size in KB

    Returns:
        bool: True if compression was successful, False otherwise
    """
    try:
        # Check if file exists and is an image
        if not os.path.exists(image_path):
            return False

        # Check current file size
        current_size = os.path.getsize(image_path)
        if current_size <= max_size_kb * 1024:
            return True  # Already small enough

        # Open the image
        img = Image.open(image_path)
        img_format = img.format if img.format else 'PNG'

        # If we get here, we couldn't compress enough with quality reduction alone
        # Try resizing the image
        width, height = img.size
        scale_factor = (max_size_kb * 1024) / current_size

        # print(f'Scale factor: {scale_factor}')
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        temp_buffer = BytesIO()
        resized_img.save(temp_buffer, format=img_format,
                         quality=25, optimize=True)
        temp_size = temp_buffer.getbuffer().nbytes

        while temp_size > max_size_kb * 1024:
            # print(f'Scale factor: {scale_factor}')
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            temp_buffer = BytesIO()
            resized_img.save(temp_buffer, format=img_format,
                             quality=90, optimize=True)
            temp_size = temp_buffer.getbuffer().nbytes
            scale_factor *= 0.5  # Reduce scale factor for next iteration

        resized_img.save(image_path, format=img_format,
                         quality=90, optimize=True)
        # print(
        #     f"Compressed and resized {image_path} to {new_width}x{new_height} ({temp_size/1024:.1f}KB)")
        return True

    except Exception as e:
        print(f"Error compressing image {image_path}: {str(e)}")
        return False

def upload_to_s3(local_directory: str, bucket_name: str, s3_directory: str,
                 add_timestamp: bool, test_name: str, data_type: str, file_type: str = None) -> dict:
    """Upload files to S3 bucket preserving directory structure.

    Args:
        local_directory (str): Local directory containing files to upload
        bucket_name (str): Name of S3 bucket
        s3_directory (str): Directory prefix in S3 bucket
        add_timestamp (bool): Whether to add a timestamp to the S3 directory
        test_name (str): Name of the test to include in the S3 directory
        data_type (str): Type of data being tested (emg, spike, emg_spike)
        file_type (str, optional): Type of file (ofpc, trad)

    Returns:
        dict: Dictionary containing:
            - 's3_directory': The S3 directory path where files were uploaded
            - 'uploaded_files': List of dictionaries with file info (local_path, s3_path, s3_url)
    """
    try:
        s3_client = boto3.client('s3')
        uploaded_files = []

        # Add timestamp, test name, file type, and data type to S3 directory if requested
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if file_type:
                s3_directory = f"{s3_directory}/{timestamp}_{test_name}_{file_type}_{data_type}"
            else:
                s3_directory = f"{s3_directory}/{timestamp}_{test_name}_{data_type}"

        # Find all output files
        files_dict = find_output_files(local_directory)

        # Count total files to upload
        total_files = sum(len(files) for files in files_dict.values())
        uploaded_count = 0

        # Upload each file
        for ext, file_list in files_dict.items():
            for local_path in file_list:
                # Get path relative to local_directory
                relative_path = os.path.relpath(local_path, local_directory)
                # Create S3 path preserving structure
                s3_path = os.path.join(s3_directory, relative_path)
                # Replace backslashes with forward slashes for S3
                s3_path = s3_path.replace('\\', '/')

                # Upload the file
                uploaded_count += 1
                # print(
                #     f"Uploading {uploaded_count}/{total_files}: {local_path} to s3://{bucket_name}/{s3_path}")
                s3_client.upload_file(local_path, bucket_name, s3_path)

                # Generate S3 URL
                s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_path}"

                # Add file info to uploaded_files list
                uploaded_files.append({
                    'local_path': local_path,
                    'relative_path': relative_path,
                    's3_path': s3_path,
                    's3_url': s3_url
                })

        # Generate and upload index.html
        if uploaded_files:
            # Create index.html content with file_type and data_type info
            index_html_content = generate_index_html(
                uploaded_files, s3_directory, bucket_name, local_directory)

            # Create a temporary file for index.html
            index_html_path = os.path.join(local_directory, 'index.html')
            with open(index_html_path, 'w') as f:
                f.write(index_html_content)

            # Upload index.html to S3
            s3_path = f"{s3_directory}/index.html"
            print(f"Uploading index.html to s3://{bucket_name}/{s3_path}")
            s3_client.upload_file(index_html_path, bucket_name, s3_path, ExtraArgs={
                                  'ContentType': 'text/html'})

            # Add index.html to uploaded_files list
            s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_path}"
            uploaded_files.append({
                'local_path': index_html_path,
                'relative_path': 'index.html',
                's3_path': s3_path,
                's3_url': s3_url
            })

            # Print the URL to the index.html
            print(f"Index page available at: {s3_url}")

        print(
            f"Successfully uploaded {uploaded_count + 1} files to s3://{bucket_name}/{s3_directory}")

        return {
            's3_directory': s3_directory,
            'uploaded_files': uploaded_files
        }

    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return {'s3_directory': None, 'uploaded_files': []}

def upload_test_results(data_dir, test_type, file_type, data_type=None):
    """Upload test results to S3 bucket and generate summary

    Args:
        data_dir (str): Directory containing results to upload
        test_type (str): Type of test (spike, emg, etc.)
        file_type (str): Type of file (ofpc, trad)
        data_type (str, optional): Type of data being tested (emg, spike, emg_spike)

    Returns:
        dict: Results from upload_to_s3 function
    """
    test_name = f"{test_type}_test"
    s3_dir = f"test_outputs/{os.path.basename(data_dir)}"

    # Compress all images before uploading
    print(f"Compressing images in {data_dir} before upload...")
    image_count = 0
    compressed_count = 0

    output_files = bu.find_output_files(data_dir)
    for file_list in output_files.values():
        for file in file_list:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(data_dir, file)
                image_count += 1
                if compress_image(image_path):
                    compressed_count += 1

    if image_count > 0:
        print(
            f"Compressed {compressed_count}/{image_count} images to max 50KB")

    try:
        # Upload files to S3
        upload_results = upload_to_s3(data_dir, S3_BUCKET, s3_dir,
                                         add_timestamp=True, test_name=test_name,
                                         data_type=data_type, file_type=file_type)

        # Generate summary
        summary_file = os.path.join(
            data_dir, f"{test_type}_{file_type}_s3_summary.md")
        # summary = bu.generate_github_summary(
        #     upload_results, summary_file, bucket_name=S3_BUCKET)

        # Add index.html link to summary if available
        if upload_results and upload_results.get('s3_directory'):
            index_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{upload_results['s3_directory']}/index.html"
            # Include file_type and data_type in the summary
            data_type_str = f" ({data_type})" if data_type else ""
            index_summary = f"\n\n## {test_name} - {file_type}{data_type_str}\n\nView all files in this upload: [Index Page]({index_url})\n\n"

            # Append to summary file
            with open(summary_file, 'a') as f:
                f.write(index_summary)

            # # Append to summary string
            # summary += index_summary

        # If running in GitHub Actions, append to step summary
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
                f.write(index_summary)

        return upload_results
    except Exception as e:
        print(f'Failed to upload results to S3: {str(e)}')
        return None

def dummy_upload_test_results():
    """Upload results without running tests"""
    file_type = 'ofpc'
    data_dir = data_dirs_dict[file_type]
    test_type = 'dummy'
    upload_test_results(data_dir, test_type, file_type, data_type='dummy_data')

