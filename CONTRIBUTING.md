# Contributing to GeoRacoon

First off, thanks so much for considering a contribution - we really appreciate it.
GeoRacoon is a community effort and every bit of help counts, 
whether it's fixing a typo, reporting a bug, or adding a new feature.


## Table of Contents

- [Where to start](#where-to-start)
- [Reporting bugs](#reporting-bugs)
- [Suggesting enhancements](#suggesting-enhancements)
- [Your first code contribution](#your-first-code-contribution)
- [Testing and CI](#testing-and-ci)
- [Style guide](#style-guide)
- [Getting help](#getting-help)

---

## Where to start

Not sure where to begin? Look for issues tagged [`good first issue`](https://github.com/GeoRacoon/GeoRacoon/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), these are smaller, self-contained tasks well-suited for newcomers. Issues tagged [`help wanted`](https://github.com/GeoRacoon/GeoRacoon/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) are a good next step once you're more familiar with the codebase.

**Useful resources:**
- [Documentation](https://georacoon.readthedocs.io)
- [Issue tracker](https://github.com/GeoRacoon/GeoRacoon/issues)

---

## Reporting bugs

Before opening a new issue, please search [existing issues](https://github.com/GeoRacoon/GeoRacoon/issues) (including closed ones), 
someone may have already reported the same problem.

When you do open an issue, please include:

- A clear and descriptive title
- In the description then please feature:
  - Steps to reproduce the problem (minimal example if possible)
  - What you expected to happen vs. what actually happened
  - Your operating system, Python version, and GeoRacoon version
  - Copy any relevant error messages or tracebacks

---

## Suggesting enhancements

This is an ever evolving project, so suggestions for new features are super welcome.
For that, open an issue and describe:

- What you'd like to be able to do that isn't currently possible
- Why this would be useful (your use case)
- If you have ideas on how it could work, sketch them out

This lets us all discuss the best approach before any code is written.

---

## Your first code contribution

1. **Open an issue first**: describe the contribution you plan to make so we can align on the approach before you invest time coding.
2. **Fork** the repository: <https://github.com/GeoRacoon/GeoRacoon/fork>
3. **Set up your dev environment:**
   ```bash
   git clone https://github.com/<your-username>/GeoRacoon.git
   cd GeoRacoon
   pip install -e ".[testing]"
   ```
4. **Create a feature branch:**
   ```bash
   git checkout -b feature/my-contribution
   ```
5. **Make your changes** and add tests where appropriate.
6. **Run the tests** locally to make sure everything still works:
   ```bash
   pytest
   ```
7. **Commit and push:**
   ```bash
   git commit -m "Short description of what and why"
   git push origin feature/my-contribution
   ```
8. **Open a pull request** against `main` and link the relevant issue.

We aim to review and respond to pull requests promptly.

---

## Testing and CI

When you fork the repo, the GitHub Actions workflows come with it. Once you open a PR targeting `main`, two pipelines run automatically:

**Unit tests** (`develop.yml`): runs the full test suite across Python 3.10, 3.11, 3.12, 3.13 and 3.14 on Ubuntu. After the tests pass, a coverage report is posted as a comment on your PR so you can see how your changes affect test coverage.

**Smoke tests** (`deploy.yml`): installs GeoRacoon and runs a quick end-to-end check across Ubuntu, Fedora, macOS and Windows to catch any platform-specific installation issues.

Both pipelines skip draft PRs, so you can open a draft to share early work without triggering the full test run. Mark the PR as "Ready for review" when you want the checks to run.

To run the tests locally before pushing:
```bash
pip install -e ".[testing]"
pytest
```

---

## Style guide

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code style.
- Write descriptive variable and function names, clarity over brevity.
- Add docstrings for public functions and classes.
- Keep commits focused: one logical change per commit.

---

## Getting help

If you have a question that isn't answered by the [documentation](https://georacoon.readthedocs.io), feel free to open an issue and tag it `question`. We're happy to help.