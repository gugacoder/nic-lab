# Semantic Relationship Types Guide

```yaml
---
type: domain
tags: [semantic-relationships, user-guide, relationship-selection, implementation-guide]
created: 2025-01-22
updated: 2025-01-22
status: active
up: "[[Linking System.md]]"
related: "[[Semantic Relationships.md]], [[Methodology.md]]"
---
```

## Overview

This guide provides practical instructions for understanding, selecting, and implementing semantic relationship types in the PRP system. It serves as a reference for creating rich, meaningful connections between knowledge elements that go beyond simple linking to capture sophisticated semantic relationships.

## Quick Reference

### Hierarchical Relationship Types

| Relationship Type | Semantic Meaning | When to Use | Example |
|------------------|------------------|-------------|---------|
| `inherits` / `inherited-by` | "Child IS-A specialized version of Parent" | Child concept is specialized version | JWT Auth inherits from Auth Patterns |
| `composed-of` / `composes` | "Parent HAS-A Child as integral component" | Essential structural components | User Feature composed-of Validation Logic |
| `specializes` / `generalized-by` | "Child is specific TYPE-OF Parent" | Focused implementations with constraints | OAuth specializes Auth Backend |
| `aggregates` / `part-of` | "Parent CONTAINS Child as collection" | Organizational groupings | Auth Feature aggregates Login Task |

### Lateral Relationship Types

| Relationship Type | Semantic Meaning | When to Use | Example |
|------------------|------------------|-------------|---------|
| `similar-to` | "A and B solve similar problems similarly" | >70% functional overlap | JWT similar-to Session Auth |
| `complements` / `complemented-by` | "A and B together > A+B separately" | Synergistic enhancement | Caching complements DB Optimization |
| `alternative-to` | "A and B solve same problem differently" | Different approaches to same problem | SQL alternative-to NoSQL |
| `precedes` / `follows` | "A must be completed before B" | Sequential dependencies | Schema precedes API Implementation |

## Detailed Implementation Guide

### Hierarchical Relationships

#### Inheritance Relationships

**Purpose**: Establish IS-A relationships where child concepts automatically acquire properties from parents.

**Implementation Pattern**:
```yaml
# In child file frontmatter
inherits: "[[Parent Concept.md]]"

# In parent file frontmatter  
inherited-by: "[[Child Concept.md]]"
```

**Decision Criteria**:
- ✅ Child concept is specialized version of parent
- ✅ Child should automatically acquire parent properties
- ✅ Behavioral substitutability is required
- ❌ Child is just related to or contains parent

**Validation Rules**:
- Maximum inheritance depth: 5 levels
- No circular inheritance chains
- Child must be compatible with parent's behavioral contracts

**Examples**:
```yaml
# JWT Authentication inherits from Authentication Patterns
# File: JWT Authentication.md
inherits: "[[Authentication Patterns.md]]"

# File: Authentication Patterns.md
inherited-by: "[[JWT Authentication.md]], [[OAuth Authentication.md]]"
```

#### Composition Relationships

**Purpose**: Establish HAS-A relationships where child elements are essential structural components.

**Implementation Pattern**:
```yaml
# In parent file frontmatter
composed-of: "[[Essential Component.md]]"

# In component file frontmatter
composes: "[[Parent System.md]]"
```

**Decision Criteria**:
- ✅ Component cannot exist independently of parent
- ✅ Removal of component breaks parent functionality
- ✅ Strong coupling relationship exists
- ❌ Component can exist independently

**Validation Rules**:
- Components cannot be marked as independent alternatives
- Strong coupling must be maintained
- Maximum composition depth: 4 levels

**Examples**:
```yaml
# User Registration Feature composed of validation logic
# File: User Registration Feature.md
composed-of: "[[User Validation Logic.md]], [[Email Verification.md]]"

# File: User Validation Logic.md
composes: "[[User Registration Feature.md]]"
```

#### Specialization Relationships

**Purpose**: Establish TYPE-OF relationships for focused implementations with additional constraints.

**Implementation Pattern**:
```yaml
# In specialized file frontmatter
specializes: "[[General Concept.md]]"

# In general file frontmatter
generalized-by: "[[Specialized Implementation.md]]"
```

**Decision Criteria**:
- ✅ Creating focused implementation of broader pattern
- ✅ Adding constraints while maintaining core functionality
- ✅ Behavioral substitutability with enhanced capabilities
- ❌ Completely different approach or unrelated concept

**Examples**:
```yaml
# OAuth Authentication specializes Authentication Backend
# File: OAuth Authentication.md
specializes: "[[Authentication Backend.md]]"

# File: Authentication Backend.md
generalized-by: "[[OAuth Authentication.md]], [[SAML Authentication.md]]"
```

#### Aggregation Relationships

**Purpose**: Establish CONTAINS relationships for organizational groupings of independent elements.

**Implementation Pattern**:
```yaml
# In parent file frontmatter
aggregates: "[[Independent Child.md]]"

# In child file frontmatter
part-of: "[[Parent Collection.md]]"
```

**Decision Criteria**:
- ✅ Children can exist independently of parent
- ✅ Loose coupling relationship
- ✅ Organizational grouping needed
- ❌ Strong structural dependency exists

**Examples**:
```yaml
# Authentication Feature aggregates multiple independent tasks
# File: Authentication Feature.md
aggregates: "[[Login Task.md]], [[Registration Task.md]], [[Password Reset Task.md]]"

# File: Login Task.md
part-of: "[[Authentication Feature.md]]"
```

### Lateral Relationships

#### Similarity Relationships

**Purpose**: Connect concepts that share common characteristics and approaches.

**Implementation Pattern**:
```yaml
# In both related files (symmetric relationship)
similar-to: "[[Similar Concept.md]]"
```

**Decision Criteria**:
- ✅ >70% functional overlap
- ✅ Similar problem domains and approaches
- ✅ Comparable complexity and scope
- ❌ Fundamentally different approaches

**Selection Guidelines**:
- Measure functional characteristic overlap
- Compare implementation complexity
- Assess problem domain similarity
- Verify comparable scope and effort

**Examples**:
```yaml
# JWT Authentication similar to Session Authentication
# File: JWT Authentication.md
similar-to: "[[Session Authentication.md]]"

# File: Session Authentication.md  
similar-to: "[[JWT Authentication.md]]"
```

#### Complementarity Relationships

**Purpose**: Connect concepts that work synergistically together for enhanced functionality.

**Implementation Pattern**:
```yaml
# In first file frontmatter
complements: "[[Synergistic Concept.md]]"

# In second file frontmatter
complemented-by: "[[First Concept.md]]"
```

**Decision Criteria**:
- ✅ Combined use provides greater value than sum of parts
- ✅ Non-overlapping capabilities that enhance each other
- ✅ Integration creates emergent benefits
- ❌ Concepts compete or conflict with each other

**Examples**:
```yaml
# Caching complements Database Optimization
# File: Caching Strategy.md
complements: "[[Database Optimization.md]]"

# File: Database Optimization.md
complemented-by: "[[Caching Strategy.md]]"
```

#### Alternative Relationships

**Purpose**: Connect different approaches to solving the same fundamental problem.

**Implementation Pattern**:
```yaml
# In both files (symmetric relationship)
alternative-to: "[[Different Approach.md]]"
```

**Decision Criteria**:
- ✅ Address identical core problem
- ✅ Mutually exclusive implementation approaches
- ✅ Different trade-offs and constraints
- ❌ Address different problems or can be used together

**Examples**:
```yaml
# SQL Database alternative to NoSQL Database
# File: SQL Database Design.md
alternative-to: "[[NoSQL Database Design.md]]"

# File: NoSQL Database Design.md
alternative-to: "[[SQL Database Design.md]]"
```

#### Sequence Relationships

**Purpose**: Establish temporal or logical ordering with implementation dependencies.

**Implementation Pattern**:
```yaml
# In prerequisite file frontmatter
precedes: "[[Dependent Concept.md]]"

# In dependent file frontmatter
follows: "[[Prerequisite Concept.md]]"
```

**Decision Criteria**:
- ✅ Clear temporal or logical dependency
- ✅ Sequential implementation required
- ✅ Prerequisite relationship exists
- ❌ Can be implemented independently or in parallel

**Examples**:
```yaml
# Database Schema precedes API Implementation
# File: Database Schema Design.md
precedes: "[[API Implementation.md]]"

# File: API Implementation.md
follows: "[[Database Schema Design.md]]"
```

## Common Implementation Scenarios

### Scenario 1: Building a Feature Hierarchy

When organizing a complex feature with multiple components:

```yaml
# Main Feature File: User Management Feature.md
type: feature
aggregates: "[[User Registration.md]], [[User Authentication.md]]"
composed-of: "[[User Validation Rules.md]]"

# Component Task: User Registration.md  
type: task
part-of: "[[User Management Feature.md]]"
specializes: "[[User Operations.md]]"
precedes: "[[User Authentication.md]]"

# Core Logic: User Validation Rules.md
type: domain
composes: "[[User Management Feature.md]]"
similar-to: "[[Data Validation Patterns.md]]"
```

### Scenario 2: Creating Technology Alternatives

When documenting different approaches to the same problem:

```yaml
# First Approach: REST API Design.md
alternative-to: "[[GraphQL API Design.md]]"
similar-to: "[[HTTP Service Patterns.md]]"

# Second Approach: GraphQL API Design.md  
alternative-to: "[[REST API Design.md]]"
specializes: "[[API Design Patterns.md]]"

# Related Pattern: HTTP Service Patterns.md
similar-to: "[[REST API Design.md]]"
generalized-by: "[[REST API Design.md]]"
```

### Scenario 3: Building Learning Sequences

When creating implementation or learning paths:

```yaml
# Step 1: Database Design.md
precedes: "[[Backend API Setup.md]]"
specializes: "[[Data Architecture.md]]"

# Step 2: Backend API Setup.md
follows: "[[Database Design.md]]"  
precedes: "[[Frontend Integration.md]]"
composed-of: "[[Authentication Middleware.md]]"

# Step 3: Frontend Integration.md
follows: "[[Backend API Setup.md]]"
complements: "[[Backend API Setup.md]]"
```

## Validation and Quality Assurance

### Pre-Implementation Checklist

Before creating semantic relationships:

- [ ] Identify the core semantic meaning of the relationship
- [ ] Verify the relationship matches one of the defined semantic types
- [ ] Check for potential conflicts with existing relationships
- [ ] Ensure bidirectional consistency if applicable
- [ ] Validate depth limits will be respected
- [ ] Confirm the relationship enhances rather than complicates understanding

### Common Validation Errors

#### Relationship Type Mismatches
```yaml
# INCORRECT: Using inheritance for organizational grouping
inherits: "[[Authentication Feature.md]]"  # Task inheriting from feature

# CORRECT: Using aggregation for organizational grouping
part-of: "[[Authentication Feature.md]]"   # Task as part of feature collection
```

#### Conflicting Relationships
```yaml
# INCORRECT: Conflicting relationship types
similar-to: "[[JWT Authentication.md]]"
alternative-to: "[[JWT Authentication.md]]"  # Cannot be both similar and alternative

# CORRECT: Consistent relationship selection
similar-to: "[[Session Authentication.md]]"  # Both authentication approaches
alternative-to: "[[OAuth Authentication.md]]" # Different auth paradigm
```

#### Circular Dependencies
```yaml
# INCORRECT: Circular sequence relationship
# File A.md
precedes: "[[File B.md]]"
# File B.md  
precedes: "[[File A.md]]"  # Creates A → B → A cycle

# CORRECT: Linear sequence relationship
# File A.md
precedes: "[[File B.md]]"
# File B.md
follows: "[[File A.md]]"
precedes: "[[File C.md]]"
```

### Validation Commands

Use these commands to validate semantic relationship implementation:

```bash
# Validate semantic relationship type documentation
grep -r "inheritance\|composition\|specialization\|aggregation\|similarity\|complementarity" PRPs/Domains/Semantic\ Relationships.md

# Check linking system integration
grep -r "semantic.*relationship.*type\|relationship.*taxonomy" PRPs/System/Linking\ System.md

# Validate relationship types guide creation
test -f "PRPs/System/Semantic Relationship Types Guide.md" && echo "Relationship types guide created"

# Find files using semantic relationship types
grep -r "inherits\|specializes\|composed-of\|aggregates\|similar-to\|complements\|alternative-to\|precedes" PRPs/ --include="*.md"
```

## Best Practices

### Relationship Selection Principles

1. **Start with Purpose**: Begin by understanding what semantic meaning you want to capture
2. **Use Decision Frameworks**: Apply the provided decision criteria consistently  
3. **Maintain Bidirectional Consistency**: Ensure both directions of the relationship make semantic sense
4. **Respect Depth Limits**: Stay within the specified depth limits for each relationship type
5. **Validate Regularly**: Use validation commands to check for conflicts and inconsistencies

### Quality Guidelines

- **Precision over Volume**: Choose fewer, more precise relationships over many imprecise ones
- **Semantic Clarity**: Relationships should enhance rather than confuse understanding  
- **Maintenance Awareness**: Consider how relationships will need to be maintained as content evolves
- **User Experience**: Structure relationships to support efficient navigation and context discovery
- **AI Optimization**: Create relationship patterns that enable effective AI comprehension and processing

### Common Anti-Patterns to Avoid

- **Relationship Overload**: Too many relationships that create confusion rather than clarity
- **Semantic Drift**: Relationships that lose their original meaning over time
- **Inconsistent Application**: Using different relationship types for semantically identical situations
- **Circular Complexity**: Creating circular relationship patterns that prevent clear navigation
- **Validation Neglect**: Failing to validate relationships leading to inconsistent semantic networks

## Integration with DTF Framework

### Context Assembly Impact

Semantic relationships enhance DTF context assembly by:

- **Precise Context Selection**: Relationship types enable more accurate context assembly based on semantic meaning
- **Intelligent Traversal**: Semantic relationships guide traversal algorithms to include most relevant content
- **Enhanced AI Understanding**: Rich semantic relationships improve AI comprehension of knowledge relationships
- **Efficient Navigation**: Users can navigate knowledge networks following semantic rather than just structural paths

### Feature and Task Integration

- **Feature Files**: Use aggregation and composition to organize related tasks and components
- **Task Files**: Use specialization and sequence relationships to show implementation order and patterns  
- **Domain Files**: Use inheritance and similarity relationships to build knowledge hierarchies
- **System Files**: Use complementarity relationships to show synergistic system components

This guide provides comprehensive coverage of semantic relationship types implementation. For additional examples and advanced patterns, refer to the [[Semantic Relationships.md]] domain and the [[Examples/]] directory.