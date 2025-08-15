# git-commit

## Goal

Commit changes with concise, standardized, and unambiguous commit messages, following the `conventional commits` model.

## Instructions

Read $ARGUMENTS as options.

**Commit mode**: If the `-all` option is provided, enable full-commit mode; otherwise, enable stage-only mode.

**Stage validation**: If full-commit mode is enabled, add all files from the current folder and its descendants to the stage, equivalent to `git add -A .`. If stage-only mode is enabled, commit only what is already staged.

**Commit message**: Clearly state **what was changed and why**, in a way that is readable by both humans and machines, considering only changes in files already added to the staging area.

**Extra tooling**: Use TodoWrite to guide execution.

## Structure

```bash
<type>(<scope>): <direct summary of the change>

- [optional] bullets with decisions or examples
```

## Common Types

* `feat`: Adds a new feature
* `fix`: Fixes a bug
* `docs`: Documentation changes
* `style`: Code style changes (spacing, semicolons, etc.) with no functional impact
* `refactor`: Code refactoring without changing functionality
* `perf`: Performance improvements
* `test`: Adding or updating tests
* `build`: Changes that affect the build system or dependencies
* `ci`: Changes to CI configuration files and scripts
* `chore`: Other changes that don't modify src or tests
* `revert`: Reverts a previous commit

## Recommended Scopes

* `system`, `commands`, `blueprint`, `template`
* `domain`, `feature`, `task`
* `docs` (when cross-cutting)

## Examples

```bash
docs(system): convert CLAUDE.md to README.md

- Removed AI references
- Standardized with DFT structure
```

```bash
feat(feature): add comments to blog
fix(task): fix bug in JSON export
```

## Restrictions

* Do not describe the step-by-step — declare **the final outcome**
* Avoid vague terms like "adjustments" or "improvements"
* Do not include explanations outside the commit message

## Tip

If the change isn't testable, the commit should at least be **traceable and atomic**.
