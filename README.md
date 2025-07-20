# PRP Project Template

A structured documentation framework for AI-driven development using the DTF (Domains-Tasks-Features) system, optimized for Claude Code context assembly and autonomous implementation.

## Overview

Product Requirements Prompts (PRPs) is a documentation system that transforms natural language specifications into a network of interconnected markdown files. It implements a Context Engineering pipeline that enables AI systems to understand complex project requirements through distributed, atomic documentation that can be dynamically reassembled.

## Key Features

- **Natural Language Input**: Write project specifications in plain English
- **Automatic Decomposition**: Transform specs into structured Features, Tasks, and Domains
- **Context Assembly**: Dynamic reassembly of distributed documentation for AI implementation
- **Task Tracking**: Built-in severity levels (🔴🟠🟡🟢) and status workflow
- **AI-Optimized**: Designed specifically for autonomous code generation with Claude Code

## Quick Start

1. **Write Your Specification**
   ```bash
   # Edit PRPs/PROMPT.md with your project requirements
   ```

2. **Generate PRP Structure**
   ```bash
   generate-prp
   ```

3. **Execute Tasks**
   ```bash
   execute-prp "path/to/task.md"
   ```

## Project Structure

```
PRPs/
├── PROMPT.md              # Your project specification input
├── README.md              # System navigation index
├── System/                # Framework documentation
│   ├── PRP System.md      # Overview and workflow
│   ├── Methodology.md     # DTF framework details
│   ├── File Structure.md  # Organization rules
│   ├── Linking System.md  # Relationship patterns
│   ├── Management Guidelines.md  # Best practices
│   └── Templates/         # File creation templates
├── Domains/               # Reusable knowledge patterns
├── Features/              # Development efforts
└── Tasks/                 # Actionable items
```

## Documentation Types

### Domains
Reusable knowledge and context patterns that serve as building blocks for features and tasks.

### Tasks
Specific action items with:
- **Severity Levels**: 🔴 critical, 🟠 major, 🟡 medium, 🟢 minor
- **Status Tracking**: todo → in-progress → review → done
- **Context Links**: Connected to features and domains

### Features
Development efforts that group related tasks and define implementation scope.

## Frontmatter System

All files use YAML frontmatter to define relationships:

```yaml
type: domain|task|feature
tags: [category, technology, priority]
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: active|pending|completed
up: "[[Parent.md]]"              # Parent context
feature: "[[Feature.md]]"        # Associated feature
dependencies: "[[Context.md]]"   # Required knowledge
related: "[[Related.md]]"        # Additional context
```

## Context Assembly

When executing tasks, the system:
1. Extracts frontmatter links from target file
2. Follows dependency chains (depth 3)
3. Follows up chains to root
4. Includes related context (depth 1)
5. Loads complete feature context
6. Assembles in dependency order

## Commands

### `generate-prp`
Processes `PRPs/PROMPT.md` specification into linked PRP structure:
- Analyzes natural language requirements
- Creates atomic documentation files
- Establishes proper linking relationships
- Generates task priorities and features

### `execute-prp [task-file]`
Executes task with full assembled context:
- Loads task file and all dependencies
- Assembles complete implementation context
- Provides to AI for autonomous execution

## Getting Started

1. **Clone this template**
2. **Write your project specification** in `PRPs/PROMPT.md`
3. **Run `generate-prp`** to create your documentation structure
4. **Use `execute-prp`** to implement tasks with AI assistance

## Best Practices

- Keep documentation atomic and focused
- Use descriptive file names following conventions
- Maintain proper frontmatter relationships
- Update task status as work progresses
- Never rename files (breaks links)

## Example Workflow

```bash
# 1. Write your project requirements
echo "Build a todo app with user authentication..." > PRPs/PROMPT.md

# 2. Generate PRP structure
generate-prp

# 3. Review generated structure
ls PRPs/Features/
ls PRPs/Tasks/

# 4. Execute a specific task
execute-prp "PRPs/Tasks/🟡 Task 01 - Implement User Model.md"
```

## Note

This is a documentation framework template. The actual implementation of your project (as described in PROMPT.md) would require appropriate technology stacks and development tools based on your specific requirements.

## License

This template is provided as-is for use in AI-driven development workflows.