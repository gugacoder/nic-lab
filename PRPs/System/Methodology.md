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

<<<<<<< HEAD
The PRP system implements a Context Engineering pipeline that transforms specifications into distributed documentation and reassembles context dynamically for implementation. This framework incorporates modern knowledge organization principles to optimize both human cognition and AI processing effectiveness.

### Modern Knowledge Organization Integration

The methodology integrates contemporary knowledge organization theories to enhance systematic information structuring:

#### Semantic Network Principles
- **Concept Node Design**: Each PRP file serves as a semantic node with clear conceptual boundaries and typed relationships
- **Relationship Edge Optimization**: Frontmatter links preserve semantic meaning and enable efficient context traversal
- **Network Topology Management**: File structures optimized for graph traversal while maintaining human navigability
- **Semantic Density Balance**: Information distribution prevents cognitive overload while ensuring comprehensive coverage

#### Ontological Framework Application
- **Domain Modeling**: Systematic representation of knowledge areas using formal hierarchical structures
- **Property Inheritance**: Child concepts automatically inherit patterns and relationships from parent domains
- **Aximatic Constraints**: Logical rules governing valid file relationships and dependency structures
- **Conceptual Hierarchy Validation**: Automated checking that knowledge structures maintain logical consistency

#### Cognitive Load Optimization
- **Progressive Context Assembly**: Information presented in manageable chunks following dependency hierarchies
- **Intrinsic Load Management**: Complex domains decomposed into atomic units matching cognitive processing capacity
- **Extraneous Load Elimination**: Organizational structures minimize unnecessary cognitive overhead
'- **Germane Load Enhancement**: File organization facilitates pattern recognition and knowledge schema development
'''
### DTF Framework

- **Domains**: Reusable knowledge and context patterns organized as atomic knowledge units
- **Tasks**: Specific action items with severity tracking and semantic relationship preservation
- **Features**: Development efforts grouping related tasks with hierarchical inheritance patterns
=======
The PRP system implements a Context Engineering pipeline that transforms specifications into distributed documentation and reassembles context dynamically for implementation.

### DTF Framework

- **Domains**: Reusable knowledge and context patterns
- **Tasks**: Specific action items with severity tracking
- **Features**: Development efforts grouping related tasks
>>>>>>> b7b8c36256179d4c310e1684da23819f4967b005

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
<<<<<<< HEAD
7. **Apply AI context engineering optimization** to assembled context
8. **Provide** AI-optimized context to implementation AI

### AI Context Engineering Integration

The context assembly process integrates AI optimization principles from [[AI Context Engineering.md]] to maximize AI comprehension and task execution effectiveness:

#### Information Sequencing Optimization
- **Dependency-First Ordering**: Context assembly automatically orders information with prerequisites first, supporting AI reasoning chains
- **General-to-Specific Progression**: Domain overviews loaded before specific task details, providing conceptual scaffolding  
- **Problem-Solution Pairing**: Related tasks and constraints loaded alongside solutions to enable causal reasoning
- **Conceptual Building Blocks**: Information layered systematically building on previously established knowledge

#### Cognitive Load Management
- **Chunk Optimization**: Context broken into manageable sections with clear boundaries and transitions
- **Context Window Efficiency**: Redundant information eliminated while preserving essential knowledge for task completion
- **Attention Focus**: Critical information for current task highlighted through structural positioning and formatting
- **Noise Reduction**: Task-specific filtering removes information not relevant to current implementation goals

#### Pattern Recognition Enhancement
- **Consistent Structural Patterns**: Assembled context follows predictable organizational structures AI can recognize and leverage
- **Analogical Frameworks**: New concepts connected to familiar patterns and established knowledge from linked domains
- **Meta-Pattern Documentation**: Explicit pattern identification helps AI recognize and apply recurring structures

#### Inference Support Mechanisms
- **Explicit Relationship Documentation**: Semantic relationships from linked files provide clear connection reasoning
- **Causal Chain Construction**: Dependencies and prerequisites create logical reasoning scaffolds for AI understanding
- **Constraint Documentation**: Limitations and boundaries from domain files guide AI reasoning within appropriate scope
- **Example Integration**: Concrete examples from linked content support abstract concept understanding

### Enhanced Assembly Process

The AI-optimized context assembly follows this enhanced process:

1. **Context Collection**: Gather all linked content following traditional traversal rules
2. **Dependency Optimization**: Order information using dependency-first principles from AI context engineering
3. **Cognitive Load Assessment**: Evaluate information density and apply chunking strategies if needed
4. **Pattern Recognition Preparation**: Structure content using consistent patterns and meta-pattern documentation
5. **Inference Support Integration**: Ensure explicit relationships, causal chains, and reasoning scaffolds are present
6. **Final Optimization**: Apply noise reduction and attention focus techniques for task-specific optimization
=======
7. **Provide** assembled context to implementation AI
>>>>>>> b7b8c36256179d4c310e1684da23819f4967b005

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
