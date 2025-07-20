# Generate DTF Structure

Generate comprehensive DTF (Domains-Tasks-Features) documentation from initial specification with thorough research and context engineering.

## Required Reading Before Starting

- **[[PRPs/System/PRP System.md]]** - Framework overview and workflow
- **[[PRPs/System/Methodology.md]]** - Context assembly rules and examples
- **[[PRPs/System/File Structure.md]]** - Naming conventions and frontmatter requirements
- **[[PRPs/System/Management Guidelines.md]]** - Status definitions, severity levels, and best practices

## Initial File: $ARGUMENTS

## Generation Process

Read the specification file ($ARGUMENTS) first to understand what needs to be created, which domains are involved, and what features/tasks need to be implemented. The implementing AI will only get the context you create in the DTF files and their frontmatter links.

### Research Phase

- Explore existing codebase for patterns and conventions
- Search for relevant documentation and best practices for mentioned technologies  
- Identify reusable knowledge patterns for domains
- Find implementation examples and reference code
- Review existing PRPs structure for similar features

### Planning Phase

**ULTRATHINK** about the complete DTF structure and plan your approach:

- Map all domains needed and their hierarchical relationships
- Identify each distinct feature with clear scope boundaries
- Break features into atomic, executable tasks with proper severity
- Plan frontmatter linking strategy for optimal context assembly
- Identify reference examples to create in `Features/Examples/`

### Generation Phase

Create the DTF files following exact structure and conventions from templates:

#### 1. Domain Files in `PRPs/Domains/`

- **Template**: `PRPs/System/Templates/domain-template.md`
- **Naming**: `{Title}.md`
- **Content**: Comprehensive knowledge patterns and implementation approaches
- **Frontmatter**: Proper `up`, `related` links per [[System/File Structure.md]]

#### 2. Feature Files in `PRPs/Features/`

- **Template**: `PRPs/System/Templates/feature-template.md`  
- **Naming**: `{Short Description} {Subject}.md`
- **Content**: Complete implementation context, scope, dependencies, acceptance criteria
- **Examples**: Create reference files in `PRPs/Features/Examples/` as needed
- **Frontmatter**: Required `up`, `dependencies`, optional `related` links

#### 3. Task Files in `PRPs/Tasks/`

- **Template**: `PRPs/System/Templates/task-template.md`
- **Naming**: `{severity_emoji} Task {NN} - {Verb} {Description}.md`
- **Severity**: Assign appropriate level per [[System/Management Guidelines.md]]
  - 🔴 `critical`, 🟠 `major`, 🟡 `medium`, 🟢 `minor`
- **Status**: Set initial status with color coding per [[System/Management Guidelines.md]]
  - 🔵 `todo`, 🟡 `in-progress`, 🟣 `review`, 🟢 `done`, 🔴 `blocked`
- **Content**: Specific implementation steps and validation criteria
- **Frontmatter**: Required `up`, `feature`, optional `related` links

### Context Assembly Strategy

Plan frontmatter links to ensure execute-prp can assemble complete context:

- **Dependencies first**: Link to required knowledge domains
- **Up chain**: Establish clear parent-child relationships  
- **Feature relationships**: Connect tasks to parent features
- **Related connections**: Create lateral knowledge networks

Refer to [[System/Linking System.md]] for detailed assembly rules and depth limits.

### Validation Checklist

- [ ] All frontmatter links resolve to existing files
- [ ] Context assembly paths provide complete implementation context
- [ ] All specification requirements covered by generated files
- [ ] Naming conventions match [[System/File Structure.md]]
- [ ] Templates used correctly with all required fields
- [ ] Severity and status assignments follow [[System/Management Guidelines.md]]
- [ ] No circular dependencies in link structure

## Success Criteria

The generated DTF structure should enable autonomous AI implementation through execute-prp by providing complete, linked context that follows dependency chains from any task file to all necessary knowledge domains.
