#!/bin/bash

# NIC Chat System Startup Script
# This script handles installation, configuration, and execution of the NIC Chat system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD=""
VENV_DIR="$SCRIPT_DIR/.venv"
ENV_FILE="$SCRIPT_DIR/.env"
LOG_FILE="$SCRIPT_DIR/logs/startup.log"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
    log "INFO: $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    log "SUCCESS: $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING: $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR: $1"
}

# Check if Python is available
check_python() {
    print_status "Checking Python installation..."
    
    # Try different Python commands
    for cmd in python3 python python3.11 python3.10 python3.9; do
        if command -v "$cmd" >/dev/null 2>&1; then
            local version=$($cmd --version 2>&1)
            if [[ $version =~ Python\ 3\.[89]|Python\ 3\.1[0-9] ]]; then
                PYTHON_CMD="$cmd"
                print_success "Found compatible Python: $version ($cmd)"
                
                # Check if venv module is available
                if ! $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
                    print_warning "Python venv module not available"
                    
                    # Try to install python3-venv on Debian/Ubuntu systems
                    if command -v apt-get >/dev/null 2>&1; then
                        print_status "Installing python3-venv package..."
                        if sudo -n apt-get update >/dev/null 2>&1 && sudo -n apt-get install -y python3-venv >/dev/null 2>&1; then
                            print_success "python3-venv package installed"
                        else
                            print_error "Could not install python3-venv automatically"
                            print_error "Please run: sudo apt install python3-venv"
                            exit 1
                        fi
                    else
                        print_error "Python venv module not available"
                        print_error "Please install python3-venv package for your system"
                        exit 1
                    fi
                fi
                
                return 0
            fi
        fi
    done
    
    print_error "Python 3.8+ not found. Please install Python 3.8 or higher."
    exit 1
}

# Check if pip is available
check_pip() {
    print_status "Checking pip installation..."
    
    if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        print_warning "pip not found. Attempting to install..."
        
        # Try to install pip
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y python3-pip
        elif command -v yum >/dev/null 2>&1; then
            sudo yum install -y python3-pip
        elif command -v brew >/dev/null 2>&1; then
            brew install python3
        else
            print_error "Could not install pip automatically. Please install pip manually."
            exit 1
        fi
    fi
    
    print_success "pip is available"
}

# Create virtual environment
setup_virtualenv() {
    print_status "Setting up virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        print_status "Creating virtual environment..."
        
        # Try creating venv with different approaches
        if $PYTHON_CMD -m venv "$VENV_DIR" --without-pip 2>/dev/null; then
            print_success "Virtual environment created (without pip)"
        elif $PYTHON_CMD -m venv "$VENV_DIR" --system-site-packages 2>/dev/null; then
            print_success "Virtual environment created (with system site packages)"
        else
            print_error "Failed to create virtual environment"
            exit 1
        fi
    else
        print_status "Virtual environment already exists"
    fi
    
    # Check if activate script exists
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        print_warning "Virtual environment activation script missing, recreating..."
        rm -rf "$VENV_DIR"
        $PYTHON_CMD -m venv "$VENV_DIR" --upgrade-deps
        if [ ! -f "$VENV_DIR/bin/activate" ]; then
            print_error "Failed to create proper virtual environment"
            print_error "Trying alternative approach..."
            
            # Try with system site packages as fallback
            rm -rf "$VENV_DIR"
            $PYTHON_CMD -m venv "$VENV_DIR" --system-site-packages
            
            if [ ! -f "$VENV_DIR/bin/activate" ]; then
                print_error "Cannot create virtual environment. Please check Python installation."
                exit 1
            fi
        fi
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Verify activation worked
    if [ -z "$VIRTUAL_ENV" ]; then
        print_error "Virtual environment activation failed"
        exit 1
    fi
    
    # Install/upgrade pip in virtual environment if needed
    if ! python -m pip --version >/dev/null 2>&1; then
        print_status "Installing pip in virtual environment..."
        
        # Try to install pip using get-pip.py if needed
        if command -v curl >/dev/null 2>&1; then
            curl -s https://bootstrap.pypa.io/get-pip.py | python
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- https://bootstrap.pypa.io/get-pip.py | python
        else
            print_warning "pip not available and cannot install automatically"
            print_warning "Using system pip with --user flag"
        fi
    else
        print_status "Upgrading pip in virtual environment..."
        python -m pip install --upgrade pip
    fi
    
    print_success "Virtual environment activated successfully"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Install dependencies with fallback approaches
    if python -m pip install -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null; then
        print_success "Dependencies installed via pip"
    elif pip install -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null; then
        print_success "Dependencies installed via system pip"
    elif python3 -m pip install --user -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null; then
        print_success "Dependencies installed via user pip"
        print_warning "Installed to user directory, may need PYTHONPATH adjustments"
    else
        print_error "Failed to install dependencies"
        print_error "Please install manually: pip install -r requirements.txt"
        exit 1
    fi
    
    print_success "Dependencies installation completed"
}

# Check environment configuration
check_environment() {
    print_status "Checking environment configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        print_warning ".env file not found. Creating from template..."
        
        if [ -f "$SCRIPT_DIR/.env.example" ]; then
            cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
            print_warning "Please edit .env file with your configuration:"
            print_warning "  - GITLAB_URL: Your GitLab instance URL (e.g. https://gitlab.company.com)"
            print_warning "  - GITLAB_PRIVATE_TOKEN: Your GitLab API token (needs 'api' scope)"
            print_warning "  - GROQ_API_KEY: Your Groq API key (from console.groq.com)"
            print_warning ""
            print_warning "GitLab Token Setup:"
            print_warning "  1. Go to GitLab → User Settings → Access Tokens"
            print_warning "  2. Create token with 'api' scope"
            print_warning "  3. Copy token to GITLAB_PRIVATE_TOKEN in .env"
            echo
            echo -e "${YELLOW}Press Enter to continue after configuring .env file, or Ctrl+C to exit${NC}"
            read -r
        else
            print_error ".env.example template not found"
            exit 1
        fi
    fi
    
    # Validate required environment variables
    source "$ENV_FILE"
    
    missing_vars=()
    
    if [ -z "$GITLAB_URL" ]; then
        missing_vars+=("GITLAB_URL")
    fi
    
    if [ -z "$GITLAB_PRIVATE_TOKEN" ]; then
        missing_vars+=("GITLAB_PRIVATE_TOKEN")
    fi
    
    if [ -z "$GROQ_API_KEY" ]; then
        missing_vars+=("GROQ_API_KEY")
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_warning "Optional configuration missing: ${missing_vars[*]}"
        print_warning "The application will run in demo mode"
    else
        print_success "Environment configuration complete"
    fi
}

# Test application startup
test_application() {
    print_status "Testing application configuration..."
    
    # Test Python imports
    $PYTHON_CMD -c "
import sys
sys.path.append('$SCRIPT_DIR/src')
try:
    from config.settings import get_settings
    settings = get_settings()
    print(f'✅ Configuration loaded: {settings.app_name} v{settings.version}')
    
    missing = settings.validate_required_settings()
    if missing:
        print(f'⚠️  Missing optional config: {missing}')
    else:
        print('✅ All configuration present')
    
    # Test GitLab authentication components
    try:
        from utils.secrets import SecureToken, get_env_token
        from config.gitlab_config import get_gitlab_config
        
        print('✅ GitLab authentication modules loaded')
        
        # Check if GitLab is configured
        gitlab_config = get_gitlab_config()
        instances = gitlab_config.list_instances()
        if instances:
            print(f'✅ GitLab instances configured: {len(instances)}')
        else:
            print('⚠️  No GitLab instances configured (will load from environment)')
            
    except ImportError as e:
        if 'pydantic' in str(e) or 'gitlab' in str(e):
            print('⚠️  GitLab features require full dependency installation')
        else:
            print(f'⚠️  GitLab authentication warning: {e}')
    except Exception as e:
        print(f'⚠️  GitLab configuration warning: {e}')
        
except Exception as e:
    print(f'❌ Configuration error: {e}')
    sys.exit(1)
" || {
        print_error "Application configuration test failed"
        exit 1
    }
    
    print_success "Application configuration test passed"
}

# Start the application
start_application() {
    print_status "Starting NIC Chat application..."
    
    # Set environment variables
    export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
    
    # Check if port is available
    local port=8501
    if command -v lsof >/dev/null 2>&1 && lsof -i:$port >/dev/null 2>&1; then
        print_warning "Port $port is already in use. Streamlit will try to find an available port."
    fi
    
    print_success "Application starting..."
    print_status "Access the application at: http://localhost:$port"
    print_status "Press Ctrl+C to stop the application"
    echo
    
    # Start Streamlit
    cd "$SCRIPT_DIR"
    streamlit run src/app.py --server.address=0.0.0.0 --server.port=$port --server.headless=true
}

# Cleanup function
cleanup() {
    print_status "Shutting down application..."
    # Kill any background processes if needed
    # pkill -f "streamlit run"
    print_success "Application stopped"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Help function
show_help() {
    echo "NIC Chat System - Startup Script"
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --install-only      Only install dependencies, don't start the application"
    echo "  --skip-deps         Skip dependency installation"
    echo "  --dev               Start in development mode"
    echo "  --check             Check system requirements only"
    echo "  --test              Run all connection tests (GitLab, Groq, etc.)"
    echo "  --test-gitlab       Test GitLab connection only"
    echo "  --test-groq         Test Groq API connection only"
    echo "  --clean             Clean virtual environment and reinstall"
    echo
    echo "Environment:"
    echo "  Edit .env file to configure GitLab and Groq API settings"
    echo
    echo "Examples:"
    echo "  $0                  # Full setup and start"
    echo "  $0 --install-only   # Just install dependencies"
    echo "  $0 --dev            # Start in development mode"
    echo "  $0 --test           # Test all connections"
    echo "  $0 --test-gitlab    # Test only GitLab"
    echo
}

# Parse command line arguments
INSTALL_ONLY=false
SKIP_DEPS=false
DEV_MODE=false
CHECK_ONLY=false
TEST_ALL=false
TEST_GITLAB=false
TEST_GROQ=false
CLEAN_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --install-only)
            INSTALL_ONLY=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --test)
            TEST_ALL=true
            shift
            ;;
        --test-gitlab)
            TEST_GITLAB=true
            shift
            ;;
        --test-groq)
            TEST_GROQ=true
            shift
            ;;
        --clean)
            CLEAN_INSTALL=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo
    echo "🤖 NIC Chat System Startup"
    echo "=========================="
    echo "Version: 1.0.0 with GitLab Authentication"
    echo "Documentation: See PRPs/README.md for system overview"
    echo
    
    # Clean installation if requested
    if [ "$CLEAN_INSTALL" = true ]; then
        print_status "Cleaning previous installation..."
        rm -rf "$VENV_DIR"
        print_success "Cleaned virtual environment"
    fi
    
    # System checks
    check_python
    check_pip
    
    if [ "$CHECK_ONLY" = true ]; then
        print_success "System requirements check completed"
        exit 0
    fi
    
# Test GitLab connection function
test_gitlab() {
    print_status "🔗 Testing GitLab connection..."
    
    # Load environment variables
    if [ ! -f "$ENV_FILE" ]; then
        print_error ".env file not found"
        return 1
    fi
    
    # Parse .env file safely (ignore comments and handle special characters)
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Skip comments and empty lines
        [[ $key =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        
        # Remove leading/trailing whitespace
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        # Export the variable
        export "$key"="$value"
    done < "$ENV_FILE"
    
    # Check required variables
    if [ -z "$GITLAB_URL" ]; then
        print_error "GITLAB_URL not set in .env"
        return 1
    fi
    
    if [ -z "$GITLAB_PRIVATE_TOKEN" ]; then
        print_error "GITLAB_PRIVATE_TOKEN not set in .env"
        return 1
    fi
    
    # Validate URL format
    if [[ ! "$GITLAB_URL" =~ ^https?:// ]]; then
        print_error "Invalid GITLAB_URL format: $GITLAB_URL"
        return 1
    fi
    
    # Validate token length
    if [ ${#GITLAB_PRIVATE_TOKEN} -lt 20 ]; then
        print_error "GitLab token too short (${#GITLAB_PRIVATE_TOKEN} chars)"
        return 1
    fi
    
    print_success "GitLab URL: $GITLAB_URL"
    print_success "GitLab token: ${GITLAB_PRIVATE_TOKEN:0:8}***"
    
    # Test connection using curl if available
    if command -v curl >/dev/null 2>&1; then
        print_status "Testing GitLab API connection..."
        
        local api_url="${GITLAB_URL%/}/api/v4/user"
        local response_code
        
        response_code=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "PRIVATE-TOKEN: $GITLAB_PRIVATE_TOKEN" \
            --connect-timeout 10 \
            "$api_url" 2>/dev/null)
        
        case $response_code in
            200)
                print_success "GitLab API connection successful!"
                return 0
                ;;
            401)
                print_error "GitLab authentication failed (HTTP 401)"
                print_error "Check your GITLAB_PRIVATE_TOKEN"
                return 1
                ;;
            403)
                print_error "GitLab access forbidden (HTTP 403)"
                print_error "Token may not have required permissions"
                return 1
                ;;
            000)
                print_error "GitLab connection failed - cannot reach server"
                print_error "Check GITLAB_URL and network connectivity"
                return 1
                ;;
            *)
                print_error "GitLab API returned HTTP $response_code"
                return 1
                ;;
        esac
    else
        print_warning "curl not available, skipping connection test"
        print_status "Configuration appears valid"
        return 0
    fi
}

# Test Groq API connection function  
test_groq() {
    print_status "🤖 Testing Groq API connection..."
    
    # Load environment variables
    if [ ! -f "$ENV_FILE" ]; then
        print_error ".env file not found"
        return 1
    fi
    
    # Parse .env file safely (ignore comments and handle special characters)
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Skip comments and empty lines
        [[ $key =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        
        # Remove leading/trailing whitespace
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        # Export the variable
        export "$key"="$value"
    done < "$ENV_FILE"
    
    # Check required variables
    if [ -z "$GROQ_API_KEY" ]; then
        print_error "GROQ_API_KEY not set in .env"
        return 1
    fi
    
    # Validate API key format
    if [[ ! "$GROQ_API_KEY" =~ ^gsk_ ]]; then
        print_error "Invalid Groq API key format (should start with 'gsk_')"
        return 1
    fi
    
    # Validate key length
    if [ ${#GROQ_API_KEY} -lt 40 ]; then
        print_error "Groq API key too short (${#GROQ_API_KEY} chars)"
        return 1
    fi
    
    print_success "Groq API key: ${GROQ_API_KEY:0:8}***"
    
    # Test connection using curl if available
    if command -v curl >/dev/null 2>&1; then
        print_status "Testing Groq API connection..."
        
        local api_url="https://api.groq.com/openai/v1/models"
        local response_code
        
        response_code=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer $GROQ_API_KEY" \
            --connect-timeout 10 \
            "$api_url" 2>/dev/null)
        
        case $response_code in
            200)
                print_success "Groq API connection successful!"
                return 0
                ;;
            401)
                print_error "Groq API authentication failed (HTTP 401)"
                print_error "Check your GROQ_API_KEY"
                return 1
                ;;
            403)
                print_error "Groq API access forbidden (HTTP 403)"
                print_error "API key may not have required permissions"
                return 1
                ;;
            000)
                print_error "Groq API connection failed - cannot reach server"
                print_error "Check internet connectivity"
                return 1
                ;;
            *)
                print_error "Groq API returned HTTP $response_code"
                return 1
                ;;
        esac
    else
        print_warning "curl not available, skipping connection test"
        print_status "Configuration appears valid"
        return 0
    fi
}

    # Test all connections
    if [ "$TEST_ALL" = true ]; then
        print_status "Running all connection tests..."
        
        total_tests=0
        passed_tests=0
        
        # Test GitLab
        if test_gitlab; then
            passed_tests=$((passed_tests + 1))
        fi
        total_tests=$((total_tests + 1))
        
        echo
        
        # Test Groq
        if test_groq; then
            passed_tests=$((passed_tests + 1))
        fi
        total_tests=$((total_tests + 1))
        
        echo
        print_status "Overall test results: $passed_tests/$total_tests tests passed"
        
        if [ "$passed_tests" -eq "$total_tests" ]; then
            print_success "All connection tests passed successfully!"
            exit 0
        else
            print_error "Some connection tests failed"
            print_error "Please check your .env configuration and fix any issues"
            exit 1
        fi
    fi
    
    # GitLab connection test only
    if [ "$TEST_GITLAB" = true ]; then
        if test_gitlab; then
            print_success "GitLab connection test completed successfully"
            exit 0
        else
            print_error "GitLab connection test failed"
            exit 1
        fi
    fi
    
    # Groq API test only
    if [ "$TEST_GROQ" = true ]; then
        if test_groq; then
            print_success "Groq API connection test completed successfully"
            exit 0
        else
            print_error "Groq API connection test failed"
            exit 1
        fi
    fi
    
    # Setup environment
    if [ "$SKIP_DEPS" = false ]; then
        setup_virtualenv
        install_dependencies
    else
        # Just activate existing venv
        if [ -d "$VENV_DIR" ]; then
            source "$VENV_DIR/bin/activate"
        fi
    fi
    
    # Configuration
    check_environment
    test_application
    
    if [ "$INSTALL_ONLY" = true ]; then
        print_success "Installation completed. Run './run.sh' to start the application."
        exit 0
    fi
    
    # Set development mode
    if [ "$DEV_MODE" = true ]; then
        export DEBUG=true
        export ENVIRONMENT=development
        print_status "Starting in development mode"
    fi
    
    # Start application
    start_application
}

# Run main function
main "$@"