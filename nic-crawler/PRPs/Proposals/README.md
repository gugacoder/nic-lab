# Proposals

This directory contains improvement prompts for the system, ready for generating PRPs (Project Requirements and Proposals) using the Claude Code command system.

## Overview

The proposals stored here serve as templates and starting points for creating structured project requirements and proposals. Each proposal can be processed through the automated PRP generation workflow to create comprehensive project documentation.

## Usage

To generate a PRP from a proposal:

1. **Open Claude Code**:
   ```bash
   $ claude
   ```

2. **Execute the generation command**:
   ```bash
   /PRPs:generate-prp {proposal-name}
   ```

## Structure

Each proposal in this directory should contain:
- Clear problem statement
- Proposed solution approach
- Implementation considerations
- Expected outcomes

## Workflow

The PRP generation process transforms these proposals into detailed project requirements that include technical specifications, implementation plans, and success criteria.