# PRP: [Feature Name]

## Role

You are a [Specific Technical Role] with expertise in [Relevant Technologies/Domains]. Your responsibility is to implement [Feature Name] following industry best practices and the project's architectural guidelines.

**Required Expertise:**
- [Technology Stack Component 1]
- [Technology Stack Component 2]
- [Domain Knowledge Area]
- [Specific Frameworks/Libraries]

**Context Awareness:**
- Understanding of existing codebase patterns
- Knowledge of project constraints and requirements
- Familiarity with team coding standards
- Security and performance considerations

## Objective

**Primary Goal:** [Clear, specific, measurable feature deliverable]

**Success Criteria:**
- [ ] [Measurable outcome 1]
- [ ] [Measurable outcome 2]
- [ ] [Measurable outcome 3]
- [ ] [Integration requirement]
- [ ] [Performance requirement]

**Scope Boundaries:**
- **In Scope:** [What is included in this feature]
- **Out of Scope:** [What is explicitly excluded]
- **Future Considerations:** [What might be added later]

## Motivation

**Business Value:**
- [Primary business benefit]
- [User impact and experience improvement]
- [Competitive advantage or market requirement]

**Problem Statement:**
- [Current pain point or limitation]
- [User frustration or business inefficiency]
- [Technical debt or scalability issue]

**Strategic Importance:**
- [How this fits into product roadmap]
- [Urgency level and timeline drivers]
- [Dependencies on other initiatives]

**Success Metrics:**
- [Quantifiable business impact]
- [User adoption or engagement metrics]
- [Technical performance improvements]

## Context

### Technical Environment

**Architecture:**
- [System architecture pattern (e.g., microservices, monolith)]
- [Technology stack and versions]
- [Database systems and data models]
- [External services and APIs]

**Current Codebase:**
- [Relevant existing modules and their locations]
- [Code patterns and conventions to follow]
- [Existing abstractions and interfaces to use]
- [Legacy code considerations]

### Dependencies and Constraints

**Technical Dependencies:**
- [Required libraries or frameworks]
- [API integrations needed]
- [Infrastructure requirements]
- [Database schema changes]

**Business Constraints:**
- [Timeline limitations]
- [Resource availability]
- [Compliance requirements]
- [Performance requirements]

### Documentation and References

**Technical Documentation:**
- [Link to API documentation]
- [Architecture decision records]
- [Coding standards document]
- [Security guidelines]

**External References:**
- [Industry best practices]
- [Framework documentation]
- [Security standards]
- [Performance benchmarks]

### Known Gotchas and Edge Cases

**Critical Considerations:**
- [Security vulnerabilities to avoid]
- [Performance bottlenecks to prevent]
- [Data consistency issues]
- [Browser compatibility concerns]

**Edge Cases to Handle:**
- [Unusual user inputs or behaviors]
- [Error conditions and failure modes]
- [Scale and load considerations]
- [Integration failure scenarios]

## Implementation Blueprint

### Phase 1: Foundation
**Objective:** [Setup and basic structure]

**Tasks:**
1. [Specific task with clear deliverable]
   - **Input:** [What's needed to start]
   - **Output:** [Expected result]
   - **Validation:** [How to verify completion]

2. [Next task in sequence]
   - **Input:** [Dependencies from previous tasks]
   - **Output:** [Expected result]
   - **Validation:** [Verification criteria]

### Phase 2: Core Implementation
**Objective:** [Main functionality development]

**Tasks:**
1. [Core feature implementation task]
   - **Input:** [Required components]
   - **Output:** [Functional feature]
   - **Validation:** [Testing criteria]

2. [Integration task]
   - **Input:** [Components to integrate]
   - **Output:** [Integrated system]
   - **Validation:** [Integration tests]

### Phase 3: Enhancement and Optimization
**Objective:** [Polish and performance optimization]

**Tasks:**
1. [Performance optimization task]
   - **Input:** [Performance metrics]
   - **Output:** [Optimized implementation]
   - **Validation:** [Performance benchmarks]

2. [Error handling and edge cases]
   - **Input:** [Edge case scenarios]
   - **Output:** [Robust implementation]
   - **Validation:** [Error handling tests]

### Code Structure

**File Organization:**
```
[Proposed file structure]
src/
├── [module_name]/
│   ├── __init__.py
│   ├── [main_component].py
│   ├── [helper_component].py
│   └── tests/
│       ├── test_[main_component].py
│       └── test_[helper_component].py
```

**Key Components:**
- **[Component Name]**: [Responsibility and interface]
- **[Component Name]**: [Responsibility and interface]
- **[Component Name]**: [Responsibility and interface]

### Integration Points

**API Endpoints:**
- `[HTTP_METHOD] /path/to/endpoint` - [Purpose and behavior]
- `[HTTP_METHOD] /path/to/endpoint` - [Purpose and behavior]

**Data Models:**
- [Model name]: [Fields and relationships]
- [Model name]: [Fields and relationships]

**External Integrations:**
- [Service Name]: [Integration purpose and method]
- [Service Name]: [Integration purpose and method]

## Validation Loop

### Level 1: Syntax and Style
**Tools and Commands:**
```bash
# Code formatting
[formatter_command]

# Linting
[linter_command]

# Type checking
[type_checker_command]
```

**Acceptance Criteria:**
- [ ] Code passes all linting rules
- [ ] Type annotations are complete and valid
- [ ] Code formatting is consistent
- [ ] No syntax errors or warnings

### Level 2: Unit Testing
**Test Coverage Requirements:**
- Minimum 90% code coverage
- All public methods tested
- Edge cases and error conditions covered
- Mock external dependencies

**Test Commands:**
```bash
# Run unit tests
[unit_test_command]

# Coverage report
[coverage_command]
```

**Test Cases to Include:**
- [Specific test scenario 1]
- [Specific test scenario 2]
- [Edge case test scenario]
- [Error handling test scenario]

### Level 3: Integration Testing
**Integration Test Scenarios:**
- [End-to-end workflow test]
- [API integration test]
- [Database interaction test]
- [External service integration test]

**Test Commands:**
```bash
# Integration tests
[integration_test_command]

# API tests
[api_test_command]
```

### Level 4: Performance and Security
**Performance Benchmarks:**
- [Response time requirement]: < [time_limit]
- [Throughput requirement]: > [requests_per_second]
- [Memory usage]: < [memory_limit]
- [Database queries]: < [query_limit] per request

**Security Checks:**
- [ ] Input validation implemented
- [ ] Authentication/authorization verified
- [ ] SQL injection prevention
- [ ] XSS protection in place

**Validation Commands:**
```bash
# Performance testing
[performance_test_command]

# Security scanning
[security_scan_command]
```

### Acceptance Testing
**User Acceptance Criteria:**
- [ ] [User can perform action X]
- [ ] [System responds within Y seconds]
- [ ] [Error messages are clear and helpful]
- [ ] [Feature works across supported browsers/devices]

**Manual Testing Checklist:**
- [ ] [Specific user workflow test]
- [ ] [Cross-browser compatibility]
- [ ] [Mobile responsiveness]
- [ ] [Accessibility compliance]

## Additional Notes

### Security Considerations
**Critical Security Points:**
- [Authentication and authorization requirements]
- [Data encryption and protection measures]
- [Input validation and sanitization]
- [Audit logging requirements]

**Security Checklist:**
- [ ] [Specific security requirement 1]
- [ ] [Specific security requirement 2]
- [ ] [Security review completed]
- [ ] [Penetration testing passed]

### Performance Considerations
**Performance Critical Paths:**
- [High-traffic endpoint or function]
- [Database query optimization points]
- [Caching strategy implementation]
- [Resource usage optimization]

**Performance Monitoring:**
- [Metrics to track]
- [Alerting thresholds]
- [Performance degradation indicators]

### Maintenance and Extensibility
**Future Extensibility:**
- [Areas designed for future expansion]
- [Plugin or extension points]
- [Configuration flexibility]
- [API versioning considerations]

**Documentation Requirements:**
- [ ] API documentation updated
- [ ] Code comments added for complex logic
- [ ] README updated with new features
- [ ] Deployment guide updated

### Rollback and Recovery
**Rollback Strategy:**
- [Database migration rollback procedure]
- [Feature flag or toggle mechanism]
- [Deployment rollback process]
- [Data recovery procedures]

**Monitoring and Alerting:**
- [Key metrics to monitor post-deployment]
- [Error rate thresholds]
- [Alert notification procedures]
- [Health check endpoints]