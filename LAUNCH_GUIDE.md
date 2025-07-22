# NIC Chat Launch Guide

## Quick Start

The NIC Chat system with GitLab authentication is ready to launch! Use the enhanced `run.sh` script:

### 1. Basic Launch
```bash
./run.sh
```
This will:
- Check system requirements (Python 3.8+, pip)
- Create virtual environment
- Install all dependencies
- Check/create .env configuration
- Test the application
- Launch Streamlit at http://localhost:8501

### 2. First-Time Setup
```bash
./run.sh --install-only
```
This installs dependencies without starting the app, useful for initial setup.

### 3. Configuration
When running for the first time, you'll need to configure `.env`:

#### Required Settings:
- `GITLAB_URL` - Your GitLab instance URL (e.g., https://gitlab.company.com)
- `GITLAB_PRIVATE_TOKEN` - Your GitLab API token (needs 'api' scope)
- `GROQ_API_KEY` - Your Groq API key (from console.groq.com)

#### GitLab Token Setup:
1. Go to GitLab → User Settings → Access Tokens
2. Create token with 'api' scope
3. Copy token to `GITLAB_PRIVATE_TOKEN` in .env

### 4. Testing Options

#### Test System Requirements:
```bash
./run.sh --check
```

#### Test All Connections:
```bash
./run.sh --test
```
Runs all available connection tests and provides a summary.

#### Test GitLab Only:
```bash
./run.sh --test-gitlab
```
Tests only GitLab connection:
- Configuration validation (URL, token format)
- API connectivity and authentication
- Project access verification

#### Test Groq API Only:
```bash
./run.sh --test-groq
```
Tests only Groq API connection:
- API key validation (format, length)
- Model availability check
- Chat functionality test

**Note**: These tests will fail if you have invalid URLs, tokens, or missing dependencies, which is the expected behavior.

#### Development Mode:
```bash
./run.sh --dev
```
Starts with debug mode enabled.

## Available Commands

| Command | Description |
|---------|-------------|
| `./run.sh` | Full setup and launch |
| `./run.sh --help` | Show all options |
| `./run.sh --install-only` | Install dependencies only |
| `./run.sh --skip-deps` | Skip dependency installation |
| `./run.sh --dev` | Start in development mode |
| `./run.sh --check` | Check system requirements |
| `./run.sh --test-gitlab` | Test GitLab authentication |
| `./run.sh --clean` | Clean and reinstall |

## Authentication Features

The launch script now includes:

✅ **GitLab Authentication Validation** - Tests secure token handling  
✅ **Configuration Validation** - Checks GitLab and API settings  
✅ **Environment Setup** - Guided .env file creation  
✅ **Security Checks** - Validates no hardcoded tokens  
✅ **Multi-Instance Support** - Can connect to multiple GitLab instances  
✅ **Error Handling** - Clear error messages and recovery guidance  

## Troubleshooting

### Missing Dependencies
```bash
./run.sh --clean --install-only
```

### Configuration Issues
```bash
./run.sh --test-gitlab
```

### Port Already in Use
The script will automatically find an available port.

### Permission Errors
Ensure your GitLab token has 'api' scope permissions.

## Next Steps

After successful launch:

1. **Test GitLab Connection** - The app will show authentication status
2. **Browse Projects** - Access your GitLab repositories through the interface  
3. **Search Knowledge Base** - Query across your GitLab content
4. **Generate Documents** - Create DOCX/PDF documents from conversations

## Security Notes

- Never commit actual tokens to version control
- Use environment variables for all sensitive configuration
- Rotate tokens regularly
- Use minimal required permissions for GitLab tokens
- The system includes secure token handling with safe logging

---

**System Status**: ✅ Ready for launch with GitLab authentication configured
**Version**: 1.0.0 with GitLab Authentication
**Documentation**: See PRPs/README.md for detailed system architecture