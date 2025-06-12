# Contributing Guidelines

Thank you for your interest in contributing to the `force-ai` project! Please read the following guidelines before submitting a merge request.

## Getting Started

First, clone the repository to your local machine and navigate into the project directory:

```shell
git clone git@github.com:NCKU-AISLAB/force-ai.git
cd force-ai
```

Then, you can choose one of the following options to set up your development environment.

### Option 1: `uv` (recommended)

`uv` is an extremely fast Python package and project manager. If you don't have `uv` installed, check out the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) to get started.

Create a virtual environment and install the dependencies:

```shell
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt
uv run mkdocs serve
```

Open your browser and navigate to [`http://localhost:8000`](http://localhost:8000) to view the website.

### Option 2: `venv` + `pip`

Create a virtual environment and activate it:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

Open your browser and navigate to [`http://localhost:8000`](http://localhost:8000) to view the website.

## Branching Strategy

We follow the [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow) for our development process.

### Naming Conventions

- **`main`**: The default branch, protected from direct commits.
- **`feature/<name>`**: Used for new features.
- **`bugfix/<name>`**: Used for bug fixes.
- **`docs/<name>`**: Used for documentation and website updates.

Branch names should be **lowercase**, use **hyphens (****`-`****)** to separate words, and be **descriptive yet concise**. For example:

```shell
git checkout -b feature/add-cli-logging
```

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard for commit messages.

Commit messages should be clear, concise, and follow the format:

```
<type>[(optional scope)]: <subject>

[optional body]

[optional footer(s)]
```

Some common commit types include:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation updates
- `refactor:` for code refactoring
- `test:` for test-related changes
- `chore:` for maintenance tasks

Minimal example:

```shell
git commit -m "feat: add logging support"
```

More examples can be also found in [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#examples).

## Submitting a Pull Request

1. Ensure your branch is up to date with `main`:

   ```shell
   git fetch origin
   git rebase origin/main
   ```

2. Push your branch to the remote repository:

   ```shell
   git push origin feature/add-cli-logging
   ```

3. Open a **Pull Request (PR)** and provide a clear description of your changes, referencing any related issues.

4. Be responsive to reviewer feedback. All PRs will be reviewed before being merged.

Thank you for contributing to `force-ai`! 🎉
