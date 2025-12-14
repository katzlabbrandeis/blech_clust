#!/usr/bin/env python3
"""
Update README.md with the latest installation test results.

This script downloads the latest installation test results artifact from GitHub Actions
and updates the README.md file with the results.

Usage:
    python update_readme_with_test_results.py [--token GITHUB_TOKEN]

The script requires a GitHub token with repo access to download artifacts.
If not provided via --token, it will look for GITHUB_TOKEN environment variable.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path


def get_latest_workflow_run(repo, token):
    """Get the latest successful workflow run for installation_test.yml"""
    url = f"https://api.github.com/repos/{repo}/actions/workflows/installation_test.yml/runs?status=completed&per_page=1"
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if data['workflow_runs']:
                return data['workflow_runs'][0]
    except urllib.error.HTTPError as e:
        print(f"Error fetching workflow runs: {e}")
        return None


def get_artifact_download_url(repo, run_id, token):
    """Get the download URL for the installation-summary artifact"""
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            for artifact in data['artifacts']:
                if artifact['name'] == 'installation-summary':
                    return artifact['archive_download_url']
    except urllib.error.HTTPError as e:
        print(f"Error fetching artifacts: {e}")
        return None


def download_and_extract_summary(download_url, token):
    """Download and extract the summary markdown file"""
    import zipfile
    import io
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    req = urllib.request.Request(download_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
            
        # Extract the markdown file from the zip
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            with zf.open('INSTALL_TEST_RESULTS.md') as f:
                return f.read().decode('utf-8')
    except Exception as e:
        print(f"Error downloading/extracting summary: {e}")
        return None


def update_readme(summary_content):
    """Update README.md with the new summary content"""
    readme_path = Path('README.md')
    
    if not readme_path.exists():
        print("README.md not found!")
        return False
    
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    # Find the markers and replace content between them
    pattern = r'(<!-- INSTALL_TEST_RESULTS_START -->).*?(<!-- INSTALL_TEST_RESULTS_END -->)'
    replacement = f'\\1\n{summary_content}\n\\2'
    
    new_content = re.sub(pattern, replacement, readme_content, flags=re.DOTALL)
    
    if new_content == readme_content:
        print("No changes made to README.md (markers not found or content unchanged)")
        return False
    
    with open(readme_path, 'w') as f:
        f.write(new_content)
    
    print("README.md updated successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Update README with installation test results')
    parser.add_argument('--token', help='GitHub token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--repo', default='katzlabbrandeis/blech_clust', 
                       help='Repository in format owner/repo')
    args = parser.parse_args()
    
    token = args.token or os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Error: GitHub token required. Provide via --token or GITHUB_TOKEN env var")
        sys.exit(1)
    
    print(f"Fetching latest workflow run for {args.repo}...")
    workflow_run = get_latest_workflow_run(args.repo, token)
    
    if not workflow_run:
        print("No completed workflow runs found")
        sys.exit(1)
    
    run_id = workflow_run['id']
    print(f"Found workflow run #{workflow_run['run_number']} (ID: {run_id})")
    
    print("Fetching artifact download URL...")
    download_url = get_artifact_download_url(args.repo, run_id, token)
    
    if not download_url:
        print("installation-summary artifact not found")
        sys.exit(1)
    
    print("Downloading and extracting summary...")
    summary_content = download_and_extract_summary(download_url, token)
    
    if not summary_content:
        print("Failed to download summary")
        sys.exit(1)
    
    print("Updating README.md...")
    if update_readme(summary_content):
        print("\nSuccess! README.md has been updated with the latest test results.")
        print("Don't forget to commit and push the changes.")
    else:
        print("\nFailed to update README.md")
        sys.exit(1)


if __name__ == '__main__':
    main()
