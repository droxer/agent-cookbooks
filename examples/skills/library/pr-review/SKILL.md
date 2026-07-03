---
name: pr-review
description: Review a pull request diff for correctness bugs, missing tests, and style drift. Use when asked to review code changes or a PR.
---

# PR Review

Review the provided diff systematically. Report findings ordered by severity;
do not pad the review with praise or restate the diff.

## Process

1. **Understand the intent.** Read the PR title/description and infer what the
   change is supposed to do before judging how it does it.
2. **Hunt correctness bugs first.** For each hunk ask: what input or state
   makes this line wrong? Check error paths, boundary values, concurrency,
   and resource cleanup.
3. **Check test coverage.** Every behavior change needs a test that would
   fail without the change. Name the missing cases concretely.
4. **Check for drift.** Naming, comment density, and idioms should match the
   surrounding file; flag deviations only when they hurt the reader.

## Output format

For each finding: `severity (high/med/low) - file:line - one-sentence defect -
concrete failure scenario`. If nothing is wrong, say so in one line.
