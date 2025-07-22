# PRP Project Template

A structured documentation framework for AI-driven development using the DTF (Domains-Tasks-Features) system, optimized for Claude Code context assembly and autonomous implementation.

## Overview

Product Requirements Prompts (PRPs) is a documentation system that transforms natural language specifications into a network of interconnected markdown files. It implements a Context Engineering pipeline that enables AI systems to understand complex project requirements through distributed, atomic documentation that can be dynamically reassembled.

## Key Features

- **Natural Language Input**: Write project specifications in plain English
- **Automatic Decomposition**: Transform specs into structured Features, Tasks, and Domains
- **Context Assembly**: Dynamic reassembly of distributed documentation for AI implementation
- **Task Tracking**: Built-in severity levels (🔴🟠🟡🟢) and status workflow
- **AI-Optimized**: Designed specifically for autonomous code generation with Claude Code

## Quick Start

1. **Write Your Specification**
   ```bash
   # Edit PRPs/PROMPT.md with your project requirements
   ```

2. **Generate PRP Structure**
   ```bash
   generate-prp
   ```

3. **Execute Tasks**
   ```bash
   execute-prp "path/to/task.md"
   ```

## Project Structure

```
PRPs/
├── PROMPT.md              # Your project specification input
├── README.md              # System navigation index
├── System/                # Framework documentation
│   ├── PRP System.md      # Overview and workflow
│   ├── Methodology.md     # DTF framework details
│   ├── File Structure.md  # Organization rules
│   ├── Linking System.md  # Relationship patterns
│   ├── Management Guidelines.md  # Best practices
│   └── Templates/         # File creation templates
├── Domains/               # Reusable knowledge patterns
├── Features/              # Development efforts
└── Tasks/                 # Actionable items
```

## Documentation Types

### Domains
Reusable knowledge and context patterns that serve as building blocks for features and tasks.

### Tasks
Specific action items with:
- **Severity Levels**: 🔴 critical, 🟠 major, 🟡 medium, 🟢 minor
- **Status Tracking**: todo → in-progress → review → done
- **Context Links**: Connected to features and domains

### Features
Development efforts that group related tasks and define implementation scope.

## Frontmatter System

All files use YAML frontmatter to define relationships:

```yaml
type: domain|task|feature
tags: [category, technology, priority]
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: active|pending|completed
up: "[[Parent.md]]"              # Parent context
feature: "[[Feature.md]]"        # Associated feature
dependencies: "[[Context.md]]"   # Required knowledge
related: "[[Related.md]]"        # Additional context
```

## Context Assembly

When executing tasks, the system:
1. Extracts frontmatter links from target file
2. Follows dependency chains (depth 3)
3. Follows up chains to root
4. Includes related context (depth 1)
5. Loads complete feature context
6. Assembles in dependency order

## Commands

### `generate-prp`
Processes `PRPs/PROMPT.md` specification into linked PRP structure:
- Analyzes natural language requirements
- Creates atomic documentation files
- Establishes proper linking relationships
- Generates task priorities and features

### `execute-prp [task-file]`
Executes task with full assembled context:
- Loads task file and all dependencies
- Assembles complete implementation context
- Provides to AI for autonomous execution

## Getting Started

1. **Clone this template**
2. **Write your project specification** in `PRPs/PROMPT.md`
3. **Run `generate-prp`** to create your documentation structure
4. **Use `execute-prp`** to implement tasks with AI assistance

## Best Practices

- Keep documentation atomic and focused
- Use descriptive file names following conventions
- Maintain proper frontmatter relationships
- Update task status as work progresses
- Never rename files (breaks links)

## Example Workflow

```bash
# 1. Write your project requirements
echo "Build a todo app with user authentication..." > PRPs/PROMPT.md

# 2. Generate PRP structure
generate-prp

# 3. Review generated structure
ls PRPs/Features/
ls PRPs/Tasks/

# 4. Execute a specific task
execute-prp "PRPs/Tasks/🟡 Task 01 - Implement User Model.md"
```

## Note

This is a documentation framework template. The actual implementation of your project (as described in PROMPT.md) would require appropriate technology stacks and development tools based on your specific requirements.

## License

This template is provided as-is for use in AI-driven development workflows.
=======
# NIC Chat System

An AI-powered corporate knowledge assistant that integrates with self-hosted GitLab repositories to provide intelligent document search, conversation, and generation capabilities.

## 🌟 Features

- **AI-Powered Chat Interface**: Natural language conversations with corporate knowledge base
- **GitLab Integration**: Seamless connection to self-hosted GitLab repositories and wikis
- **Document Generation**: Create professional DOCX and PDF documents from conversations
- **Real-time Streaming**: Live AI responses with smooth token-by-token display
- **Session Management**: Persistent conversations across browser sessions
- **Corporate Ready**: Designed for self-hosted deployment with data sovereignty

## 🏗️ System Architecture

The NIC Chat system follows a modular architecture:

- **Frontend**: Streamlit-based web interface
- **AI Layer**: Groq API with LangChain orchestration using Llama-3.1
- **Integration Layer**: GitLab API connectivity via python-gitlab
- **Document Layer**: DOCX/PDF generation with ReportLab and python-docx
- **Knowledge Base**: Intelligent search and retrieval across GitLab repositories

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Access to a self-hosted GitLab instance
- Groq API key ([Get one here](https://console.groq.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gitlab-duno
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your actual configuration values
   ```

4. **Run the application**
   
   **Linux/macOS:**
   ```bash
   ./run.sh
   ```
   
   **Windows:**
   ```cmd
   run.bat
   ```
   
   **Manual start (if scripts don't work):**
   ```bash
   streamlit run src/app.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501` to access the application.

## ⚙️ Configuration

### Required Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# GitLab Configuration
GITLAB_URL=https://your-gitlab-instance.com
GITLAB_PRIVATE_TOKEN=your-gitlab-api-token

# Groq API Configuration  
GROQ_API_KEY=your-groq-api-key

# Application Settings
ENVIRONMENT=development
DEBUG=true
```

### GitLab Setup

1. **Create a GitLab Personal Access Token**:
   - Go to your GitLab instance → User Settings → Access Tokens
   - Create a token with `api` scope
   - Copy the token to your `.env` file

2. **Configure Repository Access**:
   - Ensure your token has access to the repositories you want to search
   - The system will automatically discover accessible projects

### Groq API Setup

1. **Get a Groq API Key**:
   - Visit [Groq Console](https://console.groq.com/)
   - Create an account and generate an API key
   - Add the key to your `.env` file

## 📁 Project Structure

```
gitlab-duno/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── config/
│   │   └── settings.py        # Configuration management
│   ├── utils/
│   │   └── session.py         # Session state utilities
│   ├── components/            # UI components (future tasks)
│   ├── ai/                    # AI and LangChain modules (future tasks)
│   ├── integrations/          # GitLab integration (future tasks)
│   └── generation/            # Document generation (future tasks)
├── tests/                     # Test suite
├── PRPs/                      # Project Requirements Prompts
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Code quality checks
flake8 src/
black src/ --check
```

### Development Mode

Set `DEBUG=true` in your `.env` file to enable:
- Detailed error messages
- Debug information in the UI
- Hot reloading on code changes
- Development logging

### Code Style

This project follows Python best practices:
- **PEP 8** style guide
- **Type hints** for better code clarity
- **Docstrings** for all public functions
- **Black** for code formatting
- **Flake8** for linting

## 🔧 Startup Scripts

The system includes automated startup scripts for easy deployment:

### Linux/macOS (`run.sh`)

```bash
# Full setup and start
./run.sh

# Available options
./run.sh --help              # Show help
./run.sh --install-only      # Just install dependencies  
./run.sh --dev               # Start in development mode
./run.sh --check             # Check system requirements only
./run.sh --clean             # Clean virtual environment and reinstall
./run.sh --skip-deps         # Skip dependency installation
```

### Windows (`run.bat`)

```cmd
# Full setup and start
run.bat

# Available options
run.bat help                 # Show help
run.bat install-only         # Just install dependencies
run.bat dev                  # Start in development mode  
run.bat check                # Check system requirements only
run.bat clean                # Clean virtual environment and reinstall
```

### What the Scripts Do

1. **System Check**: Verify Python 3.8+ and pip installation
2. **Virtual Environment**: Create and activate Python virtual environment
3. **Dependencies**: Install all required packages from requirements.txt
4. **Configuration**: Check .env file and validate settings
5. **Application Test**: Verify the application can start properly
6. **Streamlit Launch**: Start the web application on http://localhost:8501

### Manual Validation Commands

If you prefer manual setup:

```bash
# Check Python version
python3 --version

# Install dependencies
pip install -r requirements.txt

# Verify configuration
python -c "from src.config.settings import get_settings; print(get_settings().validate_required_settings())"

# Test session management
python src/utils/session.py

# Start application manually
streamlit run src/app.py
```

## 📋 Current Status

This implementation completes **Task 01 - Initialize Streamlit Application**, providing:

✅ **Core Infrastructure**:
- Streamlit application structure
- Configuration management system
- Session state handling
- Error boundaries and logging
- Component architecture foundation

✅ **User Interface**:
- Basic chat interface with message history
- Navigation system for future features
- Settings panel with preferences
- Responsive layout with sidebar

✅ **Development Ready**:
- Comprehensive configuration system
- Proper project structure
- Development and production modes
- Testing framework setup

## 🚧 Future Development

The following features will be implemented in subsequent tasks:

- **GitLab Authentication** (Task 02): Secure GitLab API integration
- **Groq API Integration** (Task 03): AI model connectivity
- **RAG Pipeline** (Task 04): LangChain-based retrieval system
- **Document Generation** (Task 06): DOCX/PDF creation
- **Advanced UI Components** (Task 07): Enhanced chat interface

## 🔒 Security Considerations

- **Environment Variables**: Sensitive data stored in `.env` (never committed)
- **Token Security**: GitLab tokens have appropriate scope limits
- **Input Validation**: All user inputs validated and sanitized
- **Error Handling**: No sensitive data exposed in error messages
- **Session Security**: Session data properly isolated

## 📚 Documentation

- **System Documentation**: Located in `PRPs/System/`
- **Feature Specifications**: Located in `PRPs/Features/`
- **Task Details**: Located in `PRPs/Tasks/`
- **Examples**: Located in `PRPs/Features/Examples/`

## 🤝 Contributing

This project follows the DTF (Domains-Tasks-Features) methodology:

1. **Research**: Review existing PRPs documentation
2. **Plan**: Use DTF structure for feature planning
3. **Implement**: Follow task specifications and acceptance criteria
4. **Validate**: Run all verification commands and tests

## 📄 License

[Add your license information here]

## 📞 Support

For questions or issues:
- Review the PRPs documentation
- Check the validation commands
- Examine the debug information in development mode

---

**Generated by the PRP System** 🤖

*This README was created following the DTF methodology for systematic software development.*
