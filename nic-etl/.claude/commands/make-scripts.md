# make-scripts

Create or update common used scripts.

**General Instructions**

- Use TodoWrite tool to create and track execution plan
- **No Batch or Powershell**: No need to generate batch/PowerShell script. Only run.sh is required.
  - For Python virtual environments, handle the folder difference: .venv/bin on Unix-based systems and .venv/Scripts on Windows.

## 1. Planning Phase

- **Project Analysis**: Read and analyze PRPs/BLUEPRINT.md for project architecture and requirements
- **State Assessment**: Read PRPs/STATE.md if present to understand current project status and dependencies
- **Technology Stack Detection**: Automatically identify the technology specifications and frameworks in use:
  - Scan for package managers (package.json, requirements.txt, pom.xml, go.mod, etc.)
  - Detect programming languages and their versions
  - Identify database systems and external services
  - Recognize containerization and deployment patterns
- **Script Requirements Analysis**: Determine what scripts need to be created or updated based on:
  - Project complexity and component count
  - Development workflow requirements
  - Distribution and deployment needs
  - Technology-specific tooling requirements
- **Script Set Generation**: Create the appropriate set of scripts tailored to the detected stack:
  - Always generate `run.sh` for development workflow
  - Generate `make-dist.sh` when distribution packaging is needed
  - Create additional utility scripts based on project requirements
  - Ensure cross-platform compatibility where applicable

## 2. Make run.sh

- Implement a comprehensive `run.sh` script
- **Single Entry Point**: `run.sh` must be the ONLY script needed to launch the complete application in development mode
- **Zero Additional Setup**: No other commands, scripts, or manual steps required after running `./run.sh`
- **Technology-Aware Setup**: Must automatically handle development requirements based on detected technology stack:
  - **Python**: Create virtual environment (venv) and run `pip install -r requirements.txt`
  - **Node.js**: Run `npm install` or `yarn install` based on lock files
  - **Java**: Handle Maven (`mvn install`) or Gradle dependencies
  - **Other stacks**: Install according to standard dependency files (composer.json, go.mod, etc.)
- **Complete Service Startup**: Launch all system components together:
  - Frontend on port 3000 (suggested default)
  - Backend API on port 8000 (suggested default)
  - Database and other required services
  - All services running concurrently
- **Fully Self-Sufficient**: Developer only needs to run `./run.sh` - the script handles all environment setup, dependency installation, and service launching automatically
- **No Batch or Powershell**: No need to generate batch/PowerShell script. Only run.sh is required.
  - For Python virtual environments, handle the folder difference: .venv/bin on Unix-based systems and .venv/Scripts on Windows.

## 3. Make make-dist.sh

**ULTRATHINK about SOFTWARE PACKAGE STRATEGIES:**

- **Distribution Manifest Analysis**: Before any packaging, create a comprehensive inventory of what constitutes the distributable application:
  - **Source Code Identification**: Scan and categorize all application files vs development artifacts
  - **Dependency Mapping**: Distinguish between runtime dependencies (must include) vs development dependencies (exclude)
  - **Asset Classification**: Identify static assets, configuration files, documentation that belong in distribution
  - **Exclusion Pattern Recognition**: Automatically detect and exclude:
    - Development artifacts (.git, node_modules source, __pycache__, .pytest_cache, coverage reports)
    - IDE configurations (.vscode, .idea, .DS_Store)
    - Build temporaries (dist/, build/, target/, .next/, .nuxt/)
    - Environment-specific files (.env, local configs, logs, temp files)
    - Test files and mock data unless explicitly needed in production

- **Smart Packaging Strategy Selection**: Analyze the application architecture to determine optimal packaging approach:
  - **Monolithic Applications**: Single executable or container with all components
  - **Microservices**: Individual service packages with orchestration files
  - **Full-Stack Applications**: Separate frontend/backend packages or unified deployment
  - **Libraries/Frameworks**: Package as reusable components with proper metadata

- **Build Context Isolation**: Never copy entire project directory - instead:
  - **Selective File Collection**: Create staging area with only distribution-required files
  - **Dependency Resolution**: Install only production dependencies in clean environment
  - **Build Artifact Generation**: Compile/transpile source code before packaging
  - **Configuration Templating**: Generate production-ready configs from templates

- **Technology-Specific Packaging Intelligence**:
  - **Python**: Distinguish between source distribution (wheel) vs binary distribution (executable)
  - **Node.js**: Separate client-side bundles from server-side code, handle native dependencies
  - **Java**: Optimize JAR size, handle classpath dependencies, consider modular JDK features
  - **Docker**: Multi-stage builds, layer optimization, security scanning, minimal base images
  - **Static Sites**: Asset optimization, CDN preparation, cache headers configuration

- **Distribution Validation Pipeline**: Before finalizing package:
  - **Dependency Verification**: Ensure all runtime dependencies are included and compatible
  - **Functionality Testing**: Run smoke tests on packaged application
  - **Security Scanning**: Check for vulnerabilities in dependencies and configurations
  - **Size Optimization**: Minimize package size while maintaining functionality
  - **Installation Testing**: Verify package installs correctly on clean target environment

- Implement a comprehensive `make-dist.sh` script for production distribution packaging
- **Single Distribution Command**: `make-dist.sh` must be the ONLY script needed to generate production-ready distribution packages
- **Intelligent Format Detection**: Automatically analyze the technology stack and suggest optimal distribution format:
  - **Python**: Generate standalone executables (PyInstaller), wheel packages, or Docker images
  - **Node.js**: Create pkg executables, npm packages, or containerized distributions
  - **Java**: Build JAR/WAR files, native executables (GraalVM), or Docker images
  - **Go**: Compile cross-platform binaries for multiple architectures
  - **Web Apps**: Generate static builds, Docker images, or deployment-ready archives
- **Interactive Format Selection**: Present comprehensive CLI menu when multiple formats are viable:
  - List all compatible distribution formats for detected stack
  - Show format descriptions and use cases
  - Allow user selection or auto-select most appropriate format
- **Complete Build Pipeline**: Handle entire packaging workflow:
  - Clean previous builds
  - Install production dependencies
  - Run build/compilation steps
  - Generate optimized distribution package
  - Create installation instructions/scripts
  - Generate checksums and metadata
- **Cross-Platform Awareness**: Generate distributions for multiple target platforms when applicable
- **Zero Manual Intervention**: Script handles all packaging requirements automatically after format selection
- **Production Ready**: Ensure generated packages are deployment-ready with proper configuration, documentation, and installation procedures

**Considerations**

- Be careful not to perform illegal operations, such as copying the current directory into one of its subdirectories.
- Make sure to identify the files and folders that are part of the distribution, so you don't package unnecessary or unwanted content.