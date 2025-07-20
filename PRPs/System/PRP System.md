# PRP System

```yaml
---
type: domain
tags: [system, overview, framework]
created: 2025-01-20
updated: 2025-01-20
status: active
related: "[[Methodology.md]]", "[[File Structure.md]]", "[[Linking System.md]]", "[[Management Guidelines.md]]"
---
```

## Overview

Product Requirements Prompts (PRPs) is a structured documentation system for organizing development knowledge, tasks, and features using a DTF (Domains-Tasks-Features) framework.

The system implements a Context Engineering pipeline that transforms general specifications into distributed documentation and then dynamically reassembles context for AI implementation.

## Core Components

### Documentation Structure

- **[[Methodology.md]]** - DTF framework, context engineering, command workflows
- **[[File Structure.md]]** - File organization, naming conventions, frontmatter rules
- **[[Linking System.md]]** - Link relationships, context assembly, error handling
- **[[Management Guidelines.md]]** - Task management, tags, best practices

### System Workflow

#### Phase 1: generate-prp

- Input: `PRPs/PROMPT.md` specification file
- Process: Decompose into Features/Tasks/Domains with appropriate links
- Output: Network of linked atomic documents

#### Phase 2: execute-prp

- Input: Single task file
- Process: Assemble complete context following frontmatter relationships
- Output: Full implementation context for autonomous AI execution

## Quick Reference

### File Types

- **Domains**: Reusable knowledge and context patterns
- **Tasks**: Specific action items with severity and status tracking
- **Features**: Development efforts with energy classification

### Key Principles

- **Atomic Documentation**: Each file contains single, focused knowledge unit
- **Linked Context**: Files interconnected via frontmatter relationships
- **Dynamic Assembly**: Context built on-demand following link chains
- **AI-Optimized**: Structure designed for autonomous AI implementation

## Implementation Commands

- `generate-prp` - Create PRP structure from PROMPT.md specification
- `execute-prp` - Execute task with full assembled context

For detailed implementation guidance, consult the component documentation linked above.
