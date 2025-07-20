# Linking System

```yaml
---
type: domain
tags: [linking, relationships, context-assembly]
created: 2025-01-20
updated: 2025-01-20
status: active
up: "[[PRP System.md]]"
related: "[[Methodology.md]], [[File Structure.md]]"
---
```

## Link Types and Relationships

### Core Link Types

- **`up`**: Parent-child hierarchical relationship
- **`related`**: Lateral connections between similar content
- **`dependencies`**: Required prerequisites for implementation
- **`feature`**: Parent feature that task implements

## Frontmatter Linking Rules

### Feature Files MUST Link To

```yaml
up: "[[Primary Domain.md]]"           # Parent domain containing this feature
dependencies: "[[Required Context.md]]"  # Domain knowledge needed for implementation
related: "[[Similar Feature.md]]"     # Optional lateral connections
```

### Task Files MUST Link To

```yaml
up: "[[Primary Domain.md]]"           # Domain containing implementation patterns  
feature: "[[Parent Feature.md]]"      # Feature this task implements
related: "[[Related Task.md]]"        # Optional task dependencies
```

### Domain Files MUST Link To

```yaml
up: "[[Parent Domain.md]]"            # Higher-level domain (if applicable)
related: "[[Related Domain.md]]"      # Cross-cutting concerns
```

## Link Traversal Rules

### Depth Limits by Link Type

- **`dependencies`**: Follow to depth 3 maximum
- **`up`**: Follow complete chain to root (no depth limit)
- **`related`**: Follow to depth 1 only
- **`feature`**: Always include complete content (depth 1)

### Traversal Priority Order

1. **Dependencies first** - Load deepest dependencies
2. **Up chain** - Load implementation patterns and architecture
3. **Related content** - Load lateral connections
4. **Feature context** - Load parent feature
5. **Target file** - Load original task/target

## Linking Strategies

### Hierarchical Navigation

Use `up` field to define parent-child relationships:

```yaml
up: "[[Authentication Backend.md]]"  # Points to parent context
```

**Purpose**: Establishes clear knowledge hierarchy and inheritance patterns

### Cross-References  

Use `related` field for lateral connections:

```yaml
related: "[[JWT Implementation.md]]"  # Points to related content
```

**Purpose**: Creates knowledge networks without hierarchical dependency

### Dependencies

Use `dependencies` field for required prerequisites:

```yaml
dependencies: "[[Database Schema.md]]"  # Must exist before this
```

**Purpose**: Defines implementation order and required knowledge

### Feature Relationships

Use `feature` field to link tasks to parent features:

```yaml
feature: "[[Auth System Feature.md]]"  # Parent feature context
```

**Purpose**: Groups related tasks under feature umbrellas

## Context Assembly Through Links

### Link Resolution Algorithm

1. **Parse target file** frontmatter
2. **Extract all link types** (up, related, dependencies, feature)
3. **Resolve dependencies** to maximum depth 3
4. **Follow up chain** completely to root
5. **Include related content** at depth 1
6. **Load feature content** completely
7. **Deduplicate** files appearing multiple times
8. **Order by dependency** hierarchy

### Link Chain Examples

#### Simple Chain

```text
🟡 Task 05 - Add User Validation.md
├── up: [[Authentication Backend.md]]
├── feature: [[User Registration Feature.md]]
└── related: [[🟢 Task 03 - Setup Database.md]]
```

**Resolution Order**:

1. `[[User Registration Feature.md]]` (feature)
2. `[[Authentication Backend.md]]` (up chain)
3. `[[🟢 Task 03 - Setup Database.md]]` (related)
4. Target task

#### Complex Chain

```text
🔴 Task 01 - Implement JWT Middleware.md
├── feature: [[Auth System Feature.md]]
│   ├── up: [[Authentication Backend.md]]
│   │   ├── up: [[Backend Architecture.md]]
│   │   └── related: [[Security Patterns.md]]
│   └── dependencies: [[Database Schema.md]]
│       └── up: [[Database Architecture.md]]
└── up: [[Authentication Backend.md]] (deduplicated)
```

**Resolution Order**:

1. `[[Database Schema.md]]` (deepest dependency)
2. `[[Database Architecture.md]]` (dependency up chain)
3. `[[Authentication Backend.md]]` (implementation patterns)
4. `[[Backend Architecture.md]]` (architecture up chain)
5. `[[Security Patterns.md]]` (related patterns)
6. `[[Auth System Feature.md]]` (feature context)
7. Target task

## Error Handling

### Missing Links

**Problem**: Referenced file doesn't exist

```yaml
up: "[[Missing Domain.md]]"  # ERROR: File not found
```

**Actions**:

- Skip link with warning in context
- Create placeholder file if critical
- Continue assembly with available links

### Circular Dependencies

**Problem**: File A links to File B, File B links to File A

```yaml
# File A.md
dependencies: "[[File B.md]]"

# File B.md  
dependencies: "[[File A.md]]"
```

**Actions**:

- Detect cycle during traversal
- Break cycle at detection point
- Include both files once in context
- Log warning about circular dependency

### Depth Limit Exceeded

**Problem**: Link chain exceeds maximum depth

```text
dependencies: A → B → C → D → E (exceeds depth 3)
```

**Actions**:

- Truncate chain at depth limit
- Include warning in assembled context
- Load files within limit only

### Invalid Link Format

**Problem**: Malformed wiki-style links

```yaml
up: "Invalid Link Format"  # Missing [[ ]]
related: [[Missing .md extension]]
```

**Actions**:

- Skip malformed links
- Log formatting error
- Continue with valid links

## Link Validation Rules

### Pre-Assembly Validation

- [ ] All linked files exist in filesystem
- [ ] Link format follows `[[Filename.md]]` pattern
- [ ] No circular dependencies detected
- [ ] Required frontmatter fields present
- [ ] Link depth limits respected

### Post-Assembly Validation

- [ ] Context includes all reachable files
- [ ] No duplicate content in assembly
- [ ] Dependency order maintained
- [ ] All required knowledge present
- [ ] Link warnings documented
