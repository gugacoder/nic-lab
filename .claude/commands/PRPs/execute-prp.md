# Execute DTF Task

Implement a task using the DTF task file and its assembled context following the framework's context assembly rules.

## Required Reading Before Starting

- **[[PRPs/System/PRP System.md]]** - Framework overview and workflow
- **[[PRPs/System/Methodology.md]]** - Context assembly algorithm and examples
- **[[PRPs/System/Linking System.md]]** - Link traversal rules and depth limits
- **[[PRPs/System/Management Guidelines.md]]** - Status updates and validation procedures

## Task File: $ARGUMENTS

## Execution Process

### 1. Context Assembly

- Read the specified task file and extract frontmatter links
- Follow link traversal rules per [[System/Linking System.md]]:
  - `dependencies`: Follow to depth 3
  - `up`: Follow complete chain to root
  - `related`: Follow to depth 1
  - `feature`: Include complete content
- Load all linked domains, features, and related tasks
- Assemble complete implementation context in dependency order

### 2. Planning Phase

**ULTRATHINK** before execution:

- Analyze all assembled context and requirements
- Create comprehensive implementation plan addressing all criteria
- Break down complex tasks into manageable steps using todos tools
- Identify implementation patterns from linked domains
- Use TodoWrite tool to create and track implementation plan
- Extend research with web searches and codebase exploration as needed

### 3. Status Management

Update task status following [[System/Management Guidelines.md]]:

```yaml
status: 🟡 in-progress  # Update when starting implementation
```

### 4. Implementation

- Execute the planned implementation
- Follow patterns and approaches from linked domain files
- Reference feature context for scope and acceptance criteria
- Implement all required functionality per task specifications

### 5. Validation

- Run each validation command specified in task file
- Execute acceptance criteria tests
- Fix any failures using error patterns from linked domains
- Re-run validation until all tests pass
- Cross-reference with feature requirements

### 6. Completion

- Verify all acceptance criteria met
- Run final validation suite
- Update task status:

```yaml
status: 🟢 done        # Mark as completed
updated: 2025-01-20    # Update timestamp
```

- Report completion status with summary

## Context Re-Assembly

If additional context needed during implementation:

- Reference the original task file and linked domains
- Follow link chains again for updated information
- Maintain awareness of complete context assembly throughout process

## Error Handling

When validation fails:

- Consult error handling patterns from linked domains
- Review task acceptance criteria for specific requirements
- Use related task files for implementation guidance
- Retry with corrected approach

## Success Criteria

Task implementation is complete when:

- [ ] All acceptance criteria from task file met
- [ ] All validation commands pass
- [ ] Implementation follows patterns from linked domains
- [ ] Feature requirements satisfied
- [ ] Task status updated to 🟢 done
- [ ] Final validation suite passes
