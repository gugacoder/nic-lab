# Execute Troubleshoot

## **GOAL**

Apply the best resolution for a reported issue using documented troubleshooting guides.

## **CONTEXT**

When documented issues occur, this prompt enables AI to automatically apply proven solutions by referencing specific troubleshooting guide files. 
The system reads the specified guide, performs preliminary investigation, and executes the most appropriate resolution.

## **READ $ARGUMENTS AS ISSUE REPORT**

The argument should be the number, a reference, or the filename of the troubleshooting guide (e.g., `01`, `librechat-role` or `01-When-librechat-role-not-found.md`).

If no argument is provided, ask which specific troubleshooting guide should be applied.

## **APPLY RESOLUTION**

1. **Load guide:** Read the specified troubleshooting guide from `PRPs/Troubleshoot-Guides/`
2. **Investigate:** Perform preliminary investigation of the current issue to understand the specific context
3. **Select solution:** If multiple resolutions exist in the guide, determine the best approach based on the investigation
4. **Execute:** Apply the selected resolution step-by-step as documented in the guide
5. **Verify:** Confirm the resolution was successful and note any deviations from the documented process