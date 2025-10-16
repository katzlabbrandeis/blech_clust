# Documentation Maintenance Guide

This directory contains the source files for the blech_clust documentation site, built with [Quarto](https://quarto.org/).

## Documentation Structure

```
docs/
├── _quarto.yml              # Main configuration file
├── index.qmd                # Home page
├── getting-started.qmd      # Installation and setup guide
├── tutorials.qmd            # Step-by-step tutorials
├── reference/               # API reference documentation
│   ├── index.qmd           # API reference home
│   ├── core-pipeline.qmd   # Core pipeline modules
│   ├── utilities.qmd       # Utility modules
│   ├── ephys-data.qmd      # Ephys data analysis
│   ├── qa-tools.qmd        # Quality assurance tools
│   └── emg-analysis.qmd    # EMG analysis
└── .gitignore              # Ignore generated files
```

## Prerequisites

To build the documentation locally, you need:

1. **Quarto** - Install from [quarto.org](https://quarto.org/docs/get-started/)
   ```bash
   # On Ubuntu/Debian
   wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.39/quarto-1.6.39-linux-amd64.deb
   sudo dpkg -i quarto-1.6.39-linux-amd64.deb
   
   # On macOS
   brew install quarto
   
   # On Windows
   # Download installer from quarto.org
   ```

2. **Python** (optional, only if using quartodoc for auto-generation)
   ```bash
   pip install quartodoc
   ```

## Building Documentation Locally

### Quick Build

```bash
cd docs
quarto render
```

This generates the site in `docs/_site/`. Open `docs/_site/index.html` in your browser to preview.

### Live Preview

For live preview with auto-reload:

```bash
cd docs
quarto preview
```

This starts a local server (usually at http://localhost:4200) that automatically rebuilds when you save changes.

## Updating Documentation

### Editing Existing Pages

1. **Navigate to the file** you want to edit (e.g., `docs/getting-started.qmd`)
2. **Edit the Quarto Markdown** (.qmd files are similar to regular Markdown with additional features)
3. **Preview your changes** with `quarto preview`
4. **Commit and push** your changes

### Adding New Pages

1. **Create a new .qmd file** in the appropriate directory
2. **Add front matter** at the top:
   ```yaml
   ---
   title: "Your Page Title"
   ---
   ```
3. **Add the page to navigation** in `_quarto.yml`:
   ```yaml
   website:
     sidebar:
       - section: "Your Section"
         contents:
           - your-new-page.qmd
   ```
4. **Build and preview** to verify

### Updating API Reference

The API reference pages are manually maintained for better narrative quality. To update:

1. **Edit the relevant file** in `docs/reference/`
2. **Follow the existing structure**:
   - Module overview
   - Key classes/functions
   - Usage examples
   - See also links
3. **Keep examples concise** and practical
4. **Link to related pages** for better navigation

### Configuration Changes

Edit `docs/_quarto.yml` to modify:

- **Site metadata** (title, description, repo URL)
- **Navigation structure** (navbar, sidebar)
- **Theme and styling** (colors, fonts, CSS)
- **Format options** (code highlighting, TOC settings)

## Quarto Markdown Features

### Code Blocks

````markdown
```python
# Python code with syntax highlighting
def hello():
    print("Hello, world!")
```
````

### Callouts

```markdown
::: {.callout-note}
This is a note callout
:::

::: {.callout-warning}
This is a warning callout
:::

::: {.callout-tip}
This is a tip callout
:::
```

### Cross-References

```markdown
See the [Getting Started](getting-started.qmd) guide.
See the [Core Pipeline](reference/core-pipeline.qmd) documentation.
```

### Tables

```markdown
| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
```

### Math

```markdown
Inline math: $E = mc^2$

Display math:
$$
\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$
```

## Deployment

Documentation is automatically deployed to GitHub Pages via GitHub Actions.

### Automatic Deployment

When you push to the `master` branch:

1. GitHub Actions workflow (`.github/workflows/docs.yml`) triggers
2. Quarto renders the documentation
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

**Problem:** Quarto fails to render
```
Error: Could not find file...
```

**Solution:** Check file paths and ensure all referenced files exist.

---

**Problem:** YAML parsing error
```
Error: Invalid YAML...
```

**Solution:** Validate YAML syntax in `_quarto.yml` (use a YAML linter).

### Preview Issues

**Problem:** Changes not showing in preview

**Solution:** 
- Hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)
- Restart `quarto preview`
- Clear the `.quarto` cache directory

### Deployment Issues

**Problem:** GitHub Pages not updating

**Solution:**
- Check the [Actions tab](https://github.com/katzlabbrandeis/blech_clust/actions) for errors
- Ensure GitHub Pages is enabled in repository settings
- Verify the workflow has proper permissions

## Resources

- **Quarto Documentation**: https://quarto.org/docs/guide/
- **Quarto Markdown**: https://quarto.org/docs/authoring/markdown-basics.html
- **GitHub Pages**: https://docs.github.com/en/pages
- **Markdown Guide**: https://www.markdownguide.org/

## Getting Help

- **Documentation Issues**: Open an issue on [GitHub](https://github.com/katzlabbrandeis/blech_clust/issues)
- **Quarto Questions**: Check [Quarto Discussions](https://github.com/quarto-dev/quarto-cli/discussions)
- **General Questions**: Ask in the project's discussion forum or contact maintainers

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for general contribution guidelines. For documentation-specific contributions:

1. **Fork the repository**
2. **Create a branch** for your changes
3. **Make your edits** and test locally
4. **Submit a pull request** with a clear description
5. **Respond to feedback** from reviewers

Thank you for helping improve the blech_clust documentation!
