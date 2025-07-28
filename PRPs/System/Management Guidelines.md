# Management Guidelines

```yaml
---
type: domain
tags: [management, best-practices, workflow, maintenance]
created: 2025-01-20
updated: 2025-01-20
status: active
up: "[[PRP System.md]]"
related: "[[File Structure.md]], [[Linking System.md]]"
---
```

## Task Status Management

### Status Values

- **🔵 `todo`** - Not yet started, ready for implementation
- **🟡 `in-progress`** - Currently being worked on
- **🟣 `review`** - Awaiting review or approval
- **🟢 `done`** - Completed successfully
- **🔴 `blocked`** - Cannot proceed due to dependencies

### Status Update Rules

- **Status changes**: Update `status` field in frontmatter only
- **Filename stability**: Do not modify filename when status changes
- **Link preservation**: Status changes must not break existing links
- **Timestamp updates**: Update `updated` field when status changes

### Status Workflow

```text
      ┌───────────────┐
      │     todo      │
      └───────┬───────┘
              │
┌─────────────▼────────────┐
│        in-progress       │
└─────┬───────────────┬────┘
      │               │
┌─────▼─────┐  ┌──────▼─────┐
│  review   │  │  blocked   │◄──┐
└───────┬───┘  └────┬───────┘   │
        │           │           │
      ┌─▼───────────▼─┐         │
      │     done      │         │
      └───────┬───────┘         │
              │                 │
              └─────────────────┘
```

## Severity Classification

### Severity Levels

- **🔴 `critical`** - Emergency response required, system breaking issues
- **🟠 `major`** - Important but not emergency, significant impact
- **🟡 `medium`** - Standard development work, moderate impact
- **🟢 `minor`** - Low priority, cosmetic or optimization tasks

### Severity Usage

- **Visual indication**: Emoji in filename for immediate identification
- **Frontmatter storage**: Severity stored in frontmatter for querying
- **Synchronization**: Keep filename emoji and frontmatter severity aligned
- **Priority queuing**: Use severity for work prioritization

### Severity Update Process

1. Update frontmatter `severity` field
2. Update filename emoji to match
3. Update `updated` timestamp
4. Verify links remain intact

## Tag Taxonomy

### Technology Tags

```yaml
tags: [backend, frontend, database, api, security, devops, testing]
```

### Priority Tags

```yaml
tags: [urgent, high, medium, low, critical]
```

### Scope Tags

```yaml
tags: [mvp, enhancement, refactor, bug, feature, spike]
```

### Domain Tags

```yaml
tags: [auth, payment, user-management, notification, analytics]
```

### Status Tags

```yaml
tags: [active, draft, archived, deprecated]
```

## Best Practices

### File Creation Guidelines

#### Content Quality

- Provide concrete examples over abstract descriptions
- Include validation criteria for all tasks
- Link related concepts explicitly
- Update timestamps when modifying content

#### Naming Conventions

- Use Title Case for readability
- Keep descriptions concise but specific
- Include severity emoji in task names
- Maintain consistent formatting

#### Context Linking

- Establish clear hierarchies using `up` relationships
- Create lateral connections with `related` fields  
- Define explicit dependencies for features
- Enable context discovery through comprehensive tagging

### Maintenance Procedures

#### Regular Maintenance Tasks

- Review and update task statuses weekly
- Archive completed tasks quarterly  
- Validate link integrity monthly
- Update tag taxonomy as needed
- Clean orphaned files regularly

#### Content Updates

- Update `updated` timestamp for all changes
- Maintain link consistency when moving files
- Preserve historical context in archived files
- Document breaking changes in related files

#### Quality Assurance

- Verify frontmatter completeness
- Check link format consistency
- Validate severity and status alignment
- Ensure tag taxonomy compliance

## Workflow Guidelines

### Creating New Files

1. **Determine type** (Domain/Task/Feature)
2. **Copy appropriate template** from `System/Templates/`
3. **Follow naming convention** exactly
4. **Add required frontmatter** fields
5. **Link to parent** using `up` field
6. **Tag appropriately** for discoverability

### Updating Existing Files

1. **Modify content** as needed
2. **Update `updated` timestamp**
3. **Verify link integrity**
4. **Update related files** if necessary
5. **Maintain frontmatter accuracy**

### Archiving Workflow

1. **Change status** to `archived`
2. **Update related files** to remove dependencies
3. **Move to archive folder** if desired
4. **Preserve link history** for reference
5. **Update parent contexts**

## File Lifecycle Management

### Active Files

- Regular content updates
- Status tracking enabled
- Full link participation
- Tag maintenance required

### Draft Files

- Work in progress content
- Limited link exposure
- Reduced discoverability
- Temporary status

### Archived Files

- Completed work preserved
- Historical reference value
- Reduced link participation
- Read-only maintenance

### Deprecated Files

- Outdated or replaced content
- Migration path documented
- Link redirection needed
- Scheduled for removal

## Quality Standards

### Content Requirements

- Clear, actionable descriptions
- Concrete acceptance criteria
- Relevant examples included
- Proper context linking

### Structural Requirements

- Complete frontmatter fields
- Consistent naming convention
- Appropriate file placement
- Valid link formatting

### Maintenance Standards

- Regular status updates
- Link integrity verification
- Tag accuracy maintenance
- Timestamp currency

## Validation Checklist

### Pre-Creation Validation

- [ ] Appropriate template selected
- [ ] Naming convention followed
- [ ] Required frontmatter present
- [ ] Parent context identified
- [ ] Tags assigned appropriately

### Post-Creation Validation

- [ ] File saves without errors
- [ ] Links resolve correctly
- [ ] Content meets quality standards
- [ ] Discoverability enabled
- [ ] Related files updated

### Maintenance Validation

- [ ] Status reflects current state
- [ ] Links remain valid
- [ ] Tags stay relevant
- [ ] Content remains accurate
- [ ] Timestamps updated
