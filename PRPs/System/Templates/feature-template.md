# Feature Name

```yaml
---
type: feature
tags: [domain, technology, scope]
created: 2025-01-20
updated: 2025-01-20
status: todo # 🔵 todo|🟡 in-progress|🟣 review|🟢 done|🔴 blocked
up: "[[Primary Domain.md]]"
related: "[[Related Feature.md]]"
dependencies: "[[Required Context.md]]"
---
```

## Purpose

[Write one clear paragraph describing exactly what this feature accomplishes and what specific problem it solves for users.]

## Scope

[List the specific functionalities included in this feature:]

- [Specific capability 1]
- [Specific capability 2]
- [Specific user action supported]
- [Specific data operation performed]

## User Flow

[Describe the complete user interaction sequence:]

1. [Initial user action that triggers this feature]
2. [System response or processing step]
3. [Intermediate user actions if any]
4. [Final outcome user observes]

**Success State**: [Describe what user sees when feature works correctly]

**Error Handling**: [Describe how errors are presented to user]

## Data Models

[Define the specific data structures needed for this feature:]

```yaml
[# Include data model definitions in appropriate format]
[# Examples: database schemas, API models, data structures]
[# Be specific about field types, constraints, relationships]
```

## API Specification

[Define the specific endpoints/interfaces for this feature:]

```yaml
[# Include API contracts, endpoint definitions, request/response formats]
[# Examples: REST endpoints, GraphQL queries, function signatures]
[# Be specific about parameters, return types, status codes]
```

## Technical Implementation

### Core Components

[List the specific technical components that must be built:]

- **[Component Name]**: [filepath/filename.ext] - [Specific responsibility]
- **[Service Name]**: [filepath/filename.ext] - [Specific functionality]
- **[Model Name]**: [filepath/filename.ext] - [Data structure definition]

### Integration Points

[Describe how this feature connects to existing systems:]

- **[System/Component A]**: [Specific integration requirement and data exchange]
- **[System/Component B]**: [Specific API calls or service dependencies]

### Implementation Patterns

[Specify the exact patterns this feature should follow:]

- [Pattern 1]: [Reference to domain pattern and how to apply it here]
- [Pattern 2]: [Specific coding pattern or architecture to follow]
- [Pattern 3]: [Error handling or validation pattern to use]

## Examples

[Create or reference implementation examples in PRPs/Features/Examples/. For complex examples, create folder structure. For simple examples, create single files. Reference existing user-provided examples when available.]

### Implementation References

[List specific examples that demonstrate this feature:]

- **[example-folder/](Examples/example-folder/)** - [Description of complex implementation example]
- **[example-file.ext](Examples/example-file.ext)** - [Description of specific code example]
- **[user-provided-example/](Examples/user-provided-example/)** - [Reference to existing user example]

### Example Content Guidelines

[When creating new examples in Examples/ folder:]

- [Create complete, working implementations that can be executed/tested]
- [Include README.md explaining how to run/use the example]
- [Follow patterns established in dependencies domains]
- [Include error handling and edge cases]
- [Provide both success and failure scenarios]

## Error Scenarios

[Define specific error cases and handling for this feature:]

- **[Error Type 1]**: [When it occurs] → [How to handle] → [User feedback]
- **[Error Type 2]**: [When it occurs] → [How to handle] → [User feedback]
- **[Error Type 3]**: [When it occurs] → [How to handle] → [User feedback]

## Acceptance Criteria

[List specific, measurable outcomes that verify feature success:]

- [ ] [Specific functional requirement that can be tested]
- [ ] [Specific performance requirement with metrics]
- [ ] [Specific user experience requirement]
- [ ] [Specific integration requirement]
- [ ] [Specific data integrity requirement]
- [ ] [Specific error handling requirement]

## Validation

### Testing Strategy

[Describe how to verify this feature works:]

- **Unit Tests**: [Specific components to test and what to verify]
- **Integration Tests**: [Specific integrations to test and expected outcomes]
- **User Acceptance Tests**: [Specific user scenarios to validate]

### Verification Commands

```bash
[# Include specific commands to test this feature]
[# Examples: unit test commands, integration test commands]
[# API testing commands, manual verification steps]
[# Performance testing commands if applicable]
```

### Success Metrics

[Define measurable success indicators:]

- [Specific metric 1]: [Expected value or range]
- [Specific metric 2]: [Expected value or range]
- [Specific behavior]: [Expected outcome]
