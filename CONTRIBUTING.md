# Contributing to blech_clust

Thank you for considering contributing to blech_clust! We welcome contributions from the community to help improve this project.

## How to Contribute

### Reporting Issues

If you encounter any issues, please report them using the GitHub issue tracker. Provide as much detail as possible to help us understand and resolve the issue:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages or logs

### Submitting Pull Requests

1. Fork the repository and create your branch from `master`
2. If you've added code that should be tested, add tests
3. Ensure the test suite passes
4. Make sure your code lints (pre-commit hooks will run automatically)
5. Update documentation if needed
6. Issue that pull request!

### Coding Standards

- Follow the existing coding style and conventions used in the project
- Ensure your code is well-documented and includes comments where necessary
- Use meaningful variable and function names
- Keep functions focused and modular
- Add docstrings to all public functions and classes

### Commit Messages

Write clear and concise commit messages:
- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Include a brief description of the changes made
- Reference issues and pull requests when relevant

### Documentation Contributions

We welcome improvements to our documentation! See [docs/README.md](docs/README.md) for detailed instructions on:

- Building documentation locally
- Editing existing pages
- Adding new pages
- Using Quarto Markdown features
- Deployment process

**Quick start for documentation:**

```bash
# Install Quarto (if not already installed)
# See https://quarto.org/docs/get-started/

# Navigate to docs directory
cd docs

# Preview documentation with live reload
quarto preview

# Make your edits to .qmd files

# Build the site
quarto render
```

**For auto-generated API docs:**

```bash
# Generate API docs from docstrings (optional)
cd docs
quartodoc build

# Preview with generated docs
quarto preview

# Commit generated docs (use -f to override gitignore)
git add -f docs/reference/api/*.qmd
git commit -m "Update auto-generated API docs"
```

Documentation is automatically deployed to GitHub Pages when changes are merged to `master`.

**Note:** Auto-generation is disabled in CI but works locally. If you update docstrings and want the API docs published, you must generate and commit them locally.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.
