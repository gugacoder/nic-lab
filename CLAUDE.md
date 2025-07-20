# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PRP (Product Requirements Prompts) documentation system for the NIC Chat project - a corporate AI platform built on Open WebUI that interfaces with the proprietary NIC LLM. The repository uses a structured DTF (Domains-Tasks-Features) framework for organizing development knowledge and automating AI-driven implementation.

## Architecture

The PRP system implements a Context Engineering pipeline:

1. **Documentation Structure**: Three-tier organization
   - **Domains/**: Reusable knowledge and context patterns
   - **Tasks/**: Specific action items with severity tracking (🔴🟠🟡🟢)
   - **Features/**: Development efforts grouping related tasks

2. **Context Assembly**: Dynamic reassembly of distributed documentation
   - Files linked via frontmatter relationships (`up`, `feature`, `dependencies`, `related`)
   - Context assembled following dependency chains to depth 3
   - Assembly order: dependencies → up chain → related → feature → target

## Core Commands

### PRP System Commands
- `generate-prp` - Process PRPs/PROMPT.md specification into linked PRP structure
- `execute-prp [task-file]` - Execute task with full assembled context

### Development Commands
These commands are conceptual - no actual implementation exists in this template repository:
- Build: Not applicable (documentation framework only)
- Test: Not applicable (no code to test)
- Lint: Not applicable (markdown files only)

## Working with the System

### Creating New Documentation
1. Write specifications in `PRPs/PROMPT.md`
2. Run `generate-prp` to create structured PRP files
3. Use `execute-prp` with any task file for implementation

### File Naming Conventions
- Tasks: `{severity_emoji} Task {NN} - {Verb} {Short Description}.md`
  - 🔴 critical - Emergency response required
  - 🟠 major - Important but not emergency
  - 🟡 medium - Standard development work
  - 🟢 minor - Low priority tasks
- Features: `{Short Description} {Subject}.md`
- Domains: `{Title}.md`

### File Templates
Use templates in `PRPs/System/Templates/`:
- `domain-template.md` - For knowledge and context files
- `task-template.md` - For action items and bugs
- `feature-template.md` - For development efforts

## Frontmatter Structure

### Required Fields (All Types)
```yaml
type: domain|task|feature
tags: [technology, priority, category]
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: active|pending|completed|archived
```

### Type-Specific Fields

**Domain Files:**
```yaml
up: "[[Parent Domain.md]]"       # OPTIONAL
related: "[[Domain.md]]"         # OPTIONAL
```

**Task Files:**
```yaml
severity: critical|major|medium|minor  # REQUIRED
up: "[[Domain.md]]"                   # REQUIRED
feature: "[[Feature.md]]"             # REQUIRED
related: "[[Task.md]]"                # OPTIONAL
```

**Feature Files:**
```yaml
up: "[[Domain.md]]"              # REQUIRED
dependencies: "[[Context.md]]"   # REQUIRED
related: "[[Feature.md]]"        # OPTIONAL
```

## Status Management

### Task Status Workflow
- 🔵 `todo` - Not yet started
- 🟡 `in-progress` - Currently being worked on
- 🟣 `review` - Awaiting review
- 🟢 `done` - Completed successfully
- 🔴 `blocked` - Cannot proceed

Important: When updating status, only modify the frontmatter `status` field. Never change filenames as this breaks links.

## Context Assembly Rules

When processing tasks:
1. Extract frontmatter links from target file
2. Follow dependency chains (`dependencies` to depth 3)
3. Follow up chains (`up` to root)
4. Include related context (`related` to depth 1)
5. Load complete feature context
6. Assemble in dependency order for implementation

## Key Files and Directories

- `PRPs/README.md` - System navigation index
- `PRPs/PROMPT.md` - Project specification input
- `PRPs/System/` - Framework documentation
  - `PRP System.md` - Overview and workflow
  - `Methodology.md` - Context engineering details
  - `File Structure.md` - Directory organization
  - `Linking System.md` - Link types and assembly
  - `Management Guidelines.md` - Task workflows and best practices
- `PRPs/System/Templates/` - File creation templates
- `PRPs/Examples/` - Implementation patterns

## Important Notes

1. This is a **documentation framework template**, not the actual NIC Chat codebase
2. No package managers, build tools, or source code are present
3. The system is designed specifically for AI-driven development with Claude Code
4. All commands (`generate-prp`, `execute-prp`) are conceptual - implementation required
5. The actual NIC Chat project described in PROMPT.md would use:
   - Frontend: SvelteKit + TypeScript
   - Backend: FastAPI + Python
   - Database: PostgreSQL with Alembic migrations
   - Base: Open WebUI fork