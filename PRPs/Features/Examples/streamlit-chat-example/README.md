# Streamlit Chat Example

This example demonstrates a basic Streamlit chat interface implementation with message history, user input, and session state management.

## Files

- `app.py` - Main Streamlit application
- `components/chat.py` - Chat components
- `utils/session.py` - Session state utilities

## Running the Example

```bash
pip install streamlit
streamlit run app.py
```

## Key Features Demonstrated

1. **Session State Management**: Proper initialization and persistence of chat messages
2. **Message Display**: Differentiation between user and assistant messages
3. **Input Handling**: Text input with Enter key submission
4. **Scrolling**: Auto-scroll to latest message
5. **Markdown Support**: Rendering formatted messages

## Code Structure

The example follows the modular structure used in the main application:
- Components are separated into individual files
- Session state is centrally managed
- Styling is consistent with the main theme