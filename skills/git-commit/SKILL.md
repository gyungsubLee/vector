---
name: git-commit
description: Build safe, focused git commits with clear messages and minimal risk of unrelated changes. Use when Codex is asked to commit code, write commit messages, stage selected files, split work into multiple commits, or review staged vs unstaged changes before committing.
---

# Git Commit

## Overview

Use this skill to turn a dirty working tree into one or more clean commits.
Prefer precise staging, verify impact quickly, and write commit messages that explain intent.

## Workflow

1. Inspect repository state.
2. Separate the change set by intent.
3. Stage only the target files.
4. Validate staged content.
5. Commit with a structured message.
6. Re-check repository state.

## Commands

Use these commands in order unless the user asks otherwise.

```bash
git status --short
git diff --name-only
git diff --cached --name-only
```

If a commit should include only specific files, stage explicit paths:

```bash
git add path/to/file1 path/to/file2
```

Preview exactly what will be committed:

```bash
git diff --cached
git diff --cached --stat
```

Use the bundled script for a concise staged summary:

```bash
bash skills/git-commit/scripts/staged_summary.sh
```

Create the commit:

```bash
git commit -m "<type>(<scope>): <summary>"
```

## Commit Message Rules

Follow `references/commit-message-format.md`.
Use a short imperative summary that matches the staged diff.
Avoid vague summaries such as `update`, `fix stuff`, or `changes`.

## Guardrails

Do not commit unrelated changes.
Do not rewrite history unless explicitly requested.
Do not use destructive git commands unless explicitly requested.
If tests are relevant and cheap, run them before commit and include the result in the response.

## Output Requirements

After committing, report:

- commit hash
- commit subject
- staged files included
- whether tests were run

If commit is blocked, report the exact blocker and the next command to resolve it.
