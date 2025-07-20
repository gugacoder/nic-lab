# Methodology

```yaml
---
type: domain
tags: [methodology, context-engineering, workflow]
created: 2025-01-20
updated: 2025-01-20
status: active
up: "[[PRP System.md]]"
related: "[[File Structure.md]], [[Linking System.md]]"
---
```

## Context Engineering Framework

The PRP system implements a Context Engineering pipeline that transforms specifications into distributed documentation and reassembles context dynamically for implementation.

### DTF Framework

- **Domains**: Reusable knowledge and context patterns
- **Tasks**: Specific action items with severity tracking
- **Features**: Development efforts grouping related tasks

## generate-prp Command

### Input Processing

**Input**: `PRPs/PROMPT.md` specification file
**Output**: Network of linked atomic documents

### Processing Algorithm

1. **Parse** PROMPT.md for distinct features and requirements
2. **Create** one feature file per distinct functionality using `Templates/feature-template.md`
3. **Create** task files for implementation steps using severity emojis
4. **Create** domain files for reusable patterns and knowledge
5. **Link** all files via frontmatter relationships

### Feature Detection Pattern

- Each noun phrase describing functionality = potential feature
- Each verb phrase describing action = potential task
- Each knowledge area mentioned = potential domain

### Success Criteria

- [ ] All features from PROMPT.md have corresponding feature files
- [ ] All tasks have proper severity classification and numbering
- [ ] All domains contain reusable knowledge patterns
- [ ] All frontmatter links resolve to existing files
- [ ] Dependency graph is acyclic and complete

## execute-prp Command

### Input Processing

**Input**: Single task file (e.g., `🔴 Task 01 - Implement Auth.md`)
**Output**: Complete implementation with full context

### Context Assembly Algorithm

1. **Start** with target task file
2. **Extract** frontmatter links: `up`, `feature`, `dependencies`, `related`
3. **Follow** each link and extract their frontmatter links
4. **Continue** recursively to depth 3 maximum
5. **Resolve** dependency order: dependencies first, then up chain, then related
6. **Load** complete content of all linked files
7. **Provide** assembled context to implementation AI

### Link Traversal Rules

- `dependencies`: Follow to depth 3
- `up`: Follow complete chain to root
- `related`: Follow to depth 1 only
- `feature`: Always include complete content

### Assembly Order

1. Load deepest dependencies first
2. Load dependency parents
3. Load implementation patterns (`up` chain)
4. Load related context
5. Load feature context
6. Load target task last

### Success Criteria

- [ ] Target task context fully assembled
- [ ] All dependencies loaded in correct order
- [ ] Implementation follows patterns from linked domains
- [ ] Task acceptance criteria met
- [ ] No broken links or missing context

## Context Assembly Examples

### Example 1: Simple Task

**Target**: `🟡 Task 05 - Add User Validation.md`

**Frontmatter**:

```yaml
up: "[[Authentication Backend.md]]"
feature: "[[User Registration Feature.md]]"
related: "[[🟢 Task 03 - Setup Database.md]]"
```

**Assembly Result**:

1. `🟡 Task 05 - Add User Validation.md` (target)
2. `[[User Registration Feature.md]]` (feature context)
3. `[[Authentication Backend.md]]` (implementation patterns)
4. `🟢 Task 03 - Setup Database.md` (related context)
5. Follow feature dependencies (depth 1)
6. Follow up chain to root

### Example 2: Complex Dependencies

**Target**: `🔴 Task 01 - Implement JWT Middleware.md`

**Context Chain**:

```text
🔴 Task 01 - Implement JWT Middleware.md
├── feature: [[Auth System Feature.md]]
│   ├── up: [[Authentication Backend.md]]
│   │   ├── up: [[Backend Architecture.md]]
│   │   └── related: [[Security Patterns.md]]
│   └── dependencies: [[Database Schema.md]]
│       └── up: [[Database Architecture.md]]
└── up: [[Authentication Backend.md]] (already loaded)
```

**Assembly Order**:

1. `[[Database Schema.md]]` (deepest dependency)
2. `[[Database Architecture.md]]` (dependency parent)
3. `[[Authentication Backend.md]]` (implementation patterns)
4. `[[Backend Architecture.md]]` (architecture parent)
5. `[[Security Patterns.md]]` (related context)
6. `[[Auth System Feature.md]]` (feature context)
7. `🔴 Task 01 - Implement JWT Middleware.md` (target)

## Context Validation

### Processing Instructions

**When processing target task**:

1. **Parse**: Extract frontmatter from target file
2. **Traverse**: Follow all link types per depth rules
3. **Deduplicate**: Remove duplicate files from context
4. **Order**: Arrange by dependency hierarchy
5. **Assemble**: Concatenate all content in dependency order
6. **Execute**: Implement using complete assembled context

**Validation Rules**:

- Verify all linked files exist
- Check for circular dependencies
- Ensure required frontmatter fields present
- Validate link depth limits respected
