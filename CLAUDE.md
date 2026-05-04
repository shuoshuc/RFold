# Repository rules for Claude Code

## Plans and specs are not committed

Files under `docs/superpowers/plans/` and `docs/superpowers/specs/` are local
working artifacts (brainstorming notes, implementation plans, design specs).
They must never be staged, committed, or pushed.

When creating commits:
- Do not `git add` anything under those two directories.
- Do not use `git add -A` / `git add .` from the repo root without confirming
  the staged set excludes plans and specs.
- If you see plans/specs in `git status`, leave them untracked.

These paths are listed in `.gitignore`, but the rule applies even if the
ignore entry is missing — treat it as authoritative.
