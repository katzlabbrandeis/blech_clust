# Documentation Maintenance Guide

This directory contains the source files for the blech_clust documentation site, built with [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## Documentation Structure

```
docs/
├── index.md                 # Home page
├── getting-started/         # Installation and setup guides
│   ├── installation.md
│   └── quickstart.md
├── tutorials.md             # Step-by-step tutorials
├── reference/               # API reference documentation
│   ├── index.md            # API reference home
│   ├── core-pipeline.md    # Core pipeline modules
│   ├── utilities.md        # Utility modules
│   ├── ephys-data.md       # Ephys data analysis
│   ├── qa-tools.md         # Quality assurance tools
│   └── emg-analysis.md     # EMG analysis
└── .gitignore              # Ignore generated files
```

## Prerequisites

To build the documentation locally, you need:

1. **Python 3.8+**
2. **MkDocs and dependencies**:
   ```bash
   pip install -r requirements/requirements-docs.txt
   ```
   
   Or install individually:
   ```bash
   pip install mkdocs-material mkdocstrings[python] pymdown-extensions
   ```

## Building Documentation Locally

### Quick Build

```bash
# From the repository root
mkdocs build
```

This generates the site in `site/`. Open `site/index.html` in your browser to preview.

### Live Preview

For live preview with auto-reload:

```bash
# From the repository root
mkdocs serve
```

This starts a local server at [http://127.0.0.1:8000](http://127.0.0.1:8000) that automatically rebuilds when you save changes to `.md` files.

## Updating Documentation

### Editing Pages

1. **Navigate to the file** you want to edit (e.g., `docs/getting-started/installation.md`)
2. **Edit the Markdown** (standard Markdown with Material extensions)
3. **Preview your changes**:
   ```bash
   mkdocs serve
   ```
4. **Commit and push** your changes

### Adding New Pages

1. **Create a new .md file** in the appropriate directory
2. **Add the page to navigation** in `mkdocs.yml`:
   ```yaml
   nav:
     - Section Name:
         - Page Title: path/to/page.md
   ```
3. **Build and preview** to verify

### API Documentation

The API documentation uses [mkdocstrings](https://mkdocstrings.github.io/) to extract docstrings from Python code.

To document a module:

```markdown
# Module Name

::: module_name.ClassName
    options:
      show_source: true
      members:
        - method_name
```

This automatically generates documentation from the module's docstrings.

### Configuration Changes

Edit `mkdocs.yml` in the repository root to modify:

- **Site metadata** (title, description, repo URL)
- **Navigation structure** (nav section)
- **Theme and styling** (theme section)
- **Plugins and extensions** (plugins and markdown_extensions sections)

## Markdown Features

### Code Blocks

````markdown
```python
# Python code with syntax highlighting
def hello():
    print("Hello, world!")
```
````

### Admonitions

```markdown
!!! note
    This is a note admonition

!!! warning
    This is a warning admonition

!!! tip
    This is a tip admonition
```

### Tabs

```markdown
=== "Tab 1"
    Content for tab 1

=== "Tab 2"
    Content for tab 2
```

### Cross-References

```markdown
See the [Getting Started](getting-started/installation.md) guide.
See the [Core Pipeline](reference/core-pipeline.md) documentation.
```

### Tables

```markdown
| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
```

## Deployment

Documentation is automatically deployed to GitHub Pages via GitHub Actions.

### Automatic Deployment

When you push to the `master` branch:

1. GitHub Actions workflow (`.github/workflows/docs.yml`) triggers
2. MkDocs builds the documentation
3. Site is deployed to GitHub Pages
4. Available at: https://katzlabbrandeis.github.io/blech_clust/

### Manual Deployment

If needed, you can manually trigger deployment:

1. Go to the [Actions tab](https://github.com/katzlabbrandeis/blech_clust/actions)
2. Select "Build and Deploy Documentation"
3. Click "Run workflow"

## Best Practices

### Writing Style

- **Be concise** - Users want quick answers
- **Use examples** - Show, don't just tell
- **Link liberally** - Connect related content
- **Keep it current** - Update docs when code changes

### Organization

- **Logical structure** - Group related content
- **Clear hierarchy** - Use headings appropriately
- **Consistent naming** - Follow existing conventions
- **Searchable** - Use descriptive titles and headings

### Code Examples

- **Test your examples** - Ensure they actually work
- **Keep them simple** - Focus on one concept at a time
- **Add comments** - Explain non-obvious parts
- **Show output** - Include expected results when helpful

### Maintenance

- **Review regularly** - Check for outdated information
- **Fix broken links** - Test links periodically
- **Update screenshots** - Keep visuals current
- **Respond to issues** - Address documentation bugs

## Troubleshooting

### Build Errors

**Problem:** MkDocs fails to build
```
Error: Config file 'mkdocs.yml' does not exist.
```

**Solution:** Run `mkdocs build` from the repository root, not the `docs/` directory.

---

**Problem:** YAML parsing error
```
Error: Invalid YAML...
```

**Solution:** Validate YAML syntax in `mkdocs.yml` (use a YAML linter).

### Preview Issues

**Problem:** Changes not showing in preview

**Solution:**
- Hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)
- Restart `mkdocs serve`
- Check for syntax errors in your Markdown

### Deployment Issues

**Problem:** GitHub Pages not updating

**Solution:**
- Check the [Actions tab](https://github.com/katzlabbrandeis/blech_clust/actions) for errors
- Ensure GitHub Pages is enabled in repository settings
- Verify the workflow has proper permissions

## Resources

- **MkDocs Documentation**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/
- **mkdocstrings**: https://mkdocstrings.github.io/
- **GitHub Pages**: https://docs.github.com/en/pages
- **Markdown Guide**: https://www.markdownguide.org/

## Getting Help

- **Documentation Issues**: Open an issue on [GitHub](https://github.com/katzlabbrandeis/blech_clust/issues)
- **MkDocs Questions**: Check [MkDocs Discussions](https://github.com/mkdocs/mkdocs/discussions)
- **General Questions**: Ask in the project's discussion forum or contact maintainers

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for general contribution guidelines. For documentation-specific contributions:

1. **Fork the repository**
2. **Create a branch** for your changes
3. **Make your edits** and test locally with `mkdocs serve`
4. **Submit a pull request** with a clear description
5. **Respond to feedback** from reviewers

Thank you for helping improve the blech_clust documentation!
