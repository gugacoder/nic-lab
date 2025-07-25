#!/usr/bin/env python3
"""
Manual Test Script for Chat Components

This script performs basic validation of the chat components to ensure
they can be imported and instantiated without errors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_component_imports():
    """Test that all chat components can be imported successfully"""
    print("Testing component imports...")
    
    try:
        # Test message component imports
        from components.chat.message import MessageComponent, MessageData, create_message_data
        print("✅ Message component imports successful")
        
        # Test message list component imports
        from components.chat.message_list import MessageListComponent, MessageListActions
        print("✅ Message list component imports successful")
        
        # Test chat input component imports
        from components.chat.chat_input import ChatInputComponent, QuickActions
        print("✅ Chat input component imports successful")
        
        # Test chat container component imports
        from components.chat.chat_container import ChatContainer, ChatContainerConfig
        print("✅ Chat container component imports successful")
        
        # Test loading indicators imports
        from components.common.loading import LoadingIndicators, ProgressTracker
        print("✅ Loading indicators imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_message_data_creation():
    """Test MessageData object creation"""
    print("\nTesting MessageData creation...")
    
    try:
        from components.chat.message import create_message_data
        
        # Create test message
        msg = create_message_data(
            role="user",
            content="Test message content",
            metadata={"test": True}
        )
        
        assert msg.role == "user"
        assert msg.content == "Test message content"
        assert msg.metadata["test"] is True
        assert msg.id is not None
        assert msg.timestamp is not None
        
        print("✅ MessageData creation successful")
        return True
        
    except Exception as e:
        print(f"❌ MessageData creation failed: {e}")
        return False

def test_config_creation():
    """Test configuration object creation"""
    print("\nTesting configuration creation...")
    
    try:
        from components.chat.chat_container import ChatContainerConfig
        
        # Test default config
        config = ChatContainerConfig.create_default()
        assert config.enable_streaming is True
        assert config.show_quick_actions is True
        
        # Test minimal config
        minimal_config = ChatContainerConfig.create_minimal()
        assert minimal_config.show_quick_actions is False
        assert minimal_config.max_messages == 20
        
        print("✅ Configuration creation successful")
        return True
        
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False

def test_css_file_exists():
    """Test that CSS file exists and is readable"""
    print("\nTesting CSS file...")
    
    try:
        css_path = Path("src/styles/chat.css")
        
        if not css_path.exists():
            print("❌ CSS file does not exist")
            return False
        
        with open(css_path, "r") as f:
            css_content = f.read()
        
        # Basic validation that it contains expected CSS
        if ".chat-container" not in css_content:
            print("❌ CSS file missing expected classes")
            return False
        
        if ".message-content" not in css_content:
            print("❌ CSS file missing message styles")
            return False
        
        print("✅ CSS file validation successful")
        return True
        
    except Exception as e:
        print(f"❌ CSS file validation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Chat Components Validation Tests\n")
    
    tests = [
        test_component_imports,
        test_message_data_creation,
        test_config_creation,
        test_css_file_exists
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Chat components are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)