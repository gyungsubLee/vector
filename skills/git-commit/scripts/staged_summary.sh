#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository." >&2
  exit 1
fi

if git diff --cached --quiet; then
  echo "No staged changes. Stage files first with: git add <path>..."
  exit 2
fi

echo "== Staged Files =="
git diff --cached --name-status

echo

echo "== Staged Stats =="
git diff --cached --stat

echo

echo "== Suggested Subject Seed =="
# Print first changed file as a quick hint for drafting a message.
first_file=$(git diff --cached --name-only | head -n 1)
echo "change(${first_file}): summarize intent"
