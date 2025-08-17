# Generate Troubeshoot

## **GOAL**

Register a frequent issue and its most common resolutions for systematic troubleshooting documentation.

## **CONTEXT**

This prompt documents issues and their proven resolutions, creating structured troubleshooting guides that can be referenced by AI systems (via execute-troubleshoot.md) to automatically apply the most effective resolution when similar issues occur.

## **READ $ARGUMENTS AS PROMPT**

If no argument is provided, ask for clarification with a single focused question about the specific issue to be documented.

## **GENERATE TROUBLESHOOT**

1. **Analyze the issue:** Study and understand the problem presented in the input prompt
2. **Apply template:** Use `PRPs/.templates/template-troubleshoot.md` as the base structure
3. **Generate file:** Create troubleshooting documentation at:
   ```
   PRPs/Troubleshoot-Guides/{sequential-number}-When-{short-description}.md
   ```

**File naming convention:**
- Use sequential numbering (01, 02, etc.), automatically determined by checking the highest existing number in `PRPs/Troubleshoot-Guides/` and incrementing by 1
- Include concise description after "When"
- Use kebab-case for the description
- Example: `01-When-librechat-role-not-found.md`