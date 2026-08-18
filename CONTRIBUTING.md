# Contributing to GeoRacoon

First off, thanks so much for considering a contribution, we really appreciate it! :raccoon:

GeoRacoon is a community effort and every bit of help counts,
whether it's fixing a typo, reporting a bug, or adding a new feature.
(This of course also includes enhancing this very file.)

> [!IMPORTANT]
> :sailboat: Just like on a boat in Ancient Greece, we all pull an oar to propel ourselves forward.
>
> Any time you want to make edits to GeoRacoon, open a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) and never one without an existing issue it relates to!
>
> *"Why do we fall, Bruce?"* :bat:
> *"So we can raise issues, to pull ourselves up again."*
>
> To link a pull request to an issue, simply copy the issue's link into the pull request's description.
>
> Still unsure how to move forward? Check out [how to link a pull request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).

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

So you've decided to contribute, but you're not sure where to begin?

First, head over to the [issue tracker](https://github.com/GeoRacoon/GeoRacoon/issues) to see what's already there. 
Someone might already be on the same trail as you :dog:.

1. Look for issues tagged [`good first issue`](https://github.com/GeoRacoon/GeoRacoon/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), these are smaller, self-contained tasks well-suited for newcomers.
2. Check issues tagged [`help wanted`](https://github.com/GeoRacoon/GeoRacoon/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22), a good next step once you're more familiar with the codebase.
3. Have a very specific idea in mind? Jump down to [Suggesting enhancements](#suggesting-enhancements).

To get an overview of the GeoRacoon codebase and more, head to these **useful resources:**
- [Documentation](https://georacoon.readthedocs.io)
- [Issue tracker](https://github.com/GeoRacoon/GeoRacoon/issues)
- New to GitHub? [Quickstart for contributing to projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)

---

## Reporting bugs

Before opening a new issue, please search [existing issues](https://github.com/GeoRacoon/GeoRacoon/issues) (including [closed ones](https://github.com/GeoRacoon/GeoRacoon/issues?q=is%3Aissue+is%3Aclosed)). 
Someone may have already reported the same problem :bug:.

When you do open an issue, please include:

- A clear and descriptive title
- In the description, be sure to cover:
  - Steps to reproduce the problem (minimal example if possible)
  - What you expected to happen vs. what actually happened
  - Your operating system, Python version, and GeoRacoon version
  - Any relevant error messages or tracebacks

---

## Suggesting enhancements

This is an ever-evolving project, so suggestions for new features are always welcome :bulb:.

Before laying out your idea, check whether it's already been raised in [existing issues](https://github.com/GeoRacoon/GeoRacoon/issues), you can add your voice to the discussion there instead.

If nothing matches your idea yet, open a new issue and describe:

- What you'd like to be able to do that isn't currently possible
- Why this would be useful (your use case)
- If you have ideas on how it could work, sketch them out

This lets us all discuss the best approach before any code gets written.

---

## Your first code contribution

Start by opening an issue describing what you plan to do so we can align before you invest time coding.

**For small changes** (typos, docs, minor fixes) you can do everything in the browser:

1. Open or find the relevant issue on the [issue tracker](https://github.com/GeoRacoon/GeoRacoon/issues).
2. Fork the repository using the "Fork" button on GitHub.
3. Edit the file(s) directly in the GitHub web editor.
4. Open a pull request from your fork against `main` and link the issue.

**For larger contributions** that need local testing:

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/GeoRacoon.git
   cd GeoRacoon
   pip install -e ".[testing]"
   ```
2. Create a feature branch:
   ```bash
   git checkout -b feature/my-contribution
   ```
3. Make your changes and add tests where appropriate.
4. Run the tests locally:
   ```bash
   pytest
   ```
5. Push and open a pull request against `main`:
   ```bash
   git push origin feature/my-contribution
   ```

We aim to review and respond to pull requests promptly.

---

## Testing and CI

When you fork the repo, the GitHub Actions workflows come with it. Once you open a PR targeting `main`, two pipelines run automatically:

- **Unit tests** (`develop.yml`): runs the full test suite across Python 3.10, 3.11, 3.12, 3.13 and 3.14 on Ubuntu.
  Once tests pass, a coverage report is posted as a comment on your PR so you can see how your changes affect test coverage.
- **Smoke tests** (`deploy.yml`): installs GeoRacoon and runs a quick end-to-end check across Ubuntu, Fedora, macOS and Windows to catch any platform-specific installation issues.

> [!NOTE]
> Both pipelines skip draft PRs, so you can open a draft to share early work without triggering the full test run.
> Mark the PR as "Ready for review" when you want the checks to run.

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