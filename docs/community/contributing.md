# Contribute to RM-Gallery

Welcome! RM-Gallery is an open-source reward model platform. Your contributions help make AI alignment and evaluation more accessible to the community.

## Get Started with Contributing

We welcome contributions of all kinds:

- **Bug Fixes**: Identify and resolve issues
- **New Features**: Add graders, training methods, or integrations
- **Documentation**: Improve guides and examples
- **Examples**: Share practical use cases and tutorials

## Set Up Development Environment

**1. Fork and Clone**

```bash
git clone https://github.com/YOUR_USERNAME/RM-Gallery.git
cd RM-Gallery
```

**2. Install Dependencies**

```bash
pip install -e ".[dev]"
```

**3. Run Tests**

```bash
pytest tests/
```

## Follow Code Standards

### Python Style

- Use `snake_case` for functions/variables, `PascalCase` for classes
- Add **type hints** for all function inputs/outputs
- Include **docstrings** (Google/NumPy format) with args/returns

**Example:**

```python
def evaluate_grader(model_path: str, dataset: str) -> dict:
    """Evaluate a grader on a benchmark dataset.

    Args:
        model_path: Path to the grader model
        dataset: Name of evaluation dataset

    Returns:
        Dictionary with evaluation metrics
    """
    # Implementation here
    pass
```

### Quality Checklist

- Handle specific exceptions with context
- Validate inputs early
- Use context managers for resources
- Optimize for readability first

## Submit Your Contribution

**1. Create a Branch**

```bash
git checkout -b feat/your-feature-name
```

**2. Make Changes**

- Write clear, focused commits
- Add tests for new features
- Update relevant documentation

**3. Commit with Convention**

Format: `<type>(<scope>): <subject>`

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

```bash
git commit -m "feat(grader): add code quality grader"
git commit -m "fix(training): resolve memory leak in BT training"
git commit -m "docs(guide): update quickstart tutorial"
```

**Rules:**
- Use imperative mood ("Add" not "Added")
- Subject < 50 characters
- Body wraps at 72 characters

**4. Push and Open PR**

```bash
git push origin feat/your-feature-name
```

Open a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Test results (if applicable)

## Contribute Documentation

When writing docs:

- **Start with "What & Why"** (1-2 sentences)
- Use **short paragraphs** and bullet lists
- **Bold** key terms for scanning
- Include **complete examples** with `qwen3-32b` as model name
- Use verb headings (Create, Run, Analyze)

See the [Documentation Style Guide](style-guide.md) for details.

## Get Help

- 🐛 **Report Bugs**: [Open an Issue](https://github.com/modelscope/RM-Gallery/issues)
- 💡 **Propose Features**: [Start a Discussion](https://github.com/modelscope/RM-Gallery/discussions)
- 📧 **Contact Team**: Check README for communication channels

> **Tip:** Before starting major work, open an issue to discuss your approach. This prevents duplicate efforts and ensures alignment with project goals.

Thank you for contributing to RM-Gallery! 🚀

