#!/bin/bash
# .devcontainer/setup.sh or similar startup script

# Authenticate with GitHub CLI using token
echo $GITHUB_TOKEN | gh auth login --with-token

# Verify authentication
gh auth status
