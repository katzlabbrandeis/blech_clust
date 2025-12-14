# GitHub Actions Scripts

This directory contains utility scripts for GitHub Actions workflows.

## update_readme_with_test_results.py

Updates the README.md file with the latest installation test results from GitHub Actions.

### Usage

```bash
# Using environment variable for token
export GITHUB_TOKEN="your_github_token_here"
python .github/scripts/update_readme_with_test_results.py

# Or pass token directly
python .github/scripts/update_readme_with_test_results.py --token "your_github_token_here"

# For a different repository
python .github/scripts/update_readme_with_test_results.py --repo "owner/repo"
```

### Requirements

- Python 3.6+
- GitHub personal access token with `repo` scope
- Standard library only (no external dependencies)

### How It Works

1. Fetches the latest completed installation test workflow run
2. Downloads the `installation-summary` artifact
3. Extracts the test results markdown
4. Updates the README.md between the `<!-- INSTALL_TEST_RESULTS_START -->` and `<!-- INSTALL_TEST_RESULTS_END -->` markers

### Creating a GitHub Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a descriptive name (e.g., "blech_clust README updater")
4. Select the `repo` scope
5. Generate and copy the token
6. Store it securely (you won't be able to see it again)

### Automation

This script can be run:
- Manually after installation tests complete
- As part of a scheduled workflow
- As a post-test step in the installation test workflow (requires appropriate permissions)
