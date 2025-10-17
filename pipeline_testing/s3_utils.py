from PIL import Image  # noqa
from io import BytesIO  # noqa
import os
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
        upload_results = bu.upload_to_s3(data_dir, S3_BUCKET, s3_dir,
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
