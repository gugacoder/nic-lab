# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NIC ETL is a comprehensive ETL pipeline for the NIC (Núcleo de Inteligência e Conhecimento) that processes documents from GitLab repositories, creates high-quality embeddings, and stores processed data in QDrant for efficient semantic search. The project is built around a **modular Jupyter Notebook architecture** where each notebook represents a pipeline stage.

## Modular Architecture

The system follows a **notebook-centric modular architecture** where the pipeline IS the notebooks:

### 🌳 **Pipeline Structure (7 Notebooks)**
```
📚 notebooks/
├── 🚀 etl.ipynb                        # Orquestrador principal
├── 🏗️ etl-1-fundacao-preparacao.ipynb # Configuração e validação  
├── 📥 etl-2-coleta-gitlab.ipynb        # Download de documentos
├── ⚙️ etl-3-processamento-docling.ipynb # Extração de conteúdo
├── 🔪 etl-4-segmentacao-chunks.ipynb   # Segmentação de texto
├── 🧠 etl-5-geracao-embeddings.ipynb   # Geração de vetores
├── 💾 etl-6-armazenamento-qdrant.ipynb # Inserção vetorial
└── 📊 rest-api.ipynb                   # API REST endpoints
```

### 🌐 **Web Dashboard Architecture** (`src/` + `static/`)
- **FastAPI Proxy** (`src/proxy.py`) - Serves static files and proxies `/api/*` to Jupyter Kernel Gateway
- **Jupyter Kernel Gateway** - Executes REST API notebook endpoints (`notebooks/rest-api.ipynb`)
- **Static Assets** (`static/`) - HTML templates, CSS, JavaScript, and Markdown content
- **Theme System** - Light/dark themes with CSS variables and semantic classes
- **Background Job Management** - Pipeline execution independent of HTTP connections

### 📦 **Data Flow Architecture**
```
📁 pipeline-data/
├── 🗂️ documents/      # GitLab downloads
├── 🧾 processed/      # Docling extractions
├── ✂️ chunks/         # Text segments
├── 🧠 embeddings/     # Vector data
├── 📊 metadata/       # Stage outputs (JSON)
└── 🔄 checkpoints/    # Completion locks
```

## Key Design Principles

1. **Notebook-First**: Each stage is a complete, self-documenting notebook
2. **Human-Readable**: A non-programmer can understand the pipeline by reading notebooks
3. **Modular Independence**: Each notebook can be run/tested independently
4. **Safe Execution**: Dependency validation prevents out-of-order execution
5. **Transparent Data Flow**: JSON + files make data flow completely visible

## Development Commands

### Running the Web Dashboard (Recommended)

**Primary Method: Web Interface + API**
```bash
# Start the complete web dashboard system
./run-server.sh

# Access the web interface at http://localhost:5000
# - Homepage: http://localhost:5000/
# - API Documentation: http://localhost:5000/api/v1
# - Pipeline Execution: http://localhost:5000 (click "Executar Pipeline ETL")
# - Status Monitoring: http://localhost:5000/api/v1/pipelines/gitlab-qdrant/runs/last
```

**Alternative: Direct Notebook Access**
```bash
# Start Jupyter Lab for direct notebook access
jupyter lab

# Open and execute the master notebook
# File: notebooks/etl.ipynb
# Run all cells for full automation
```

### Manual Pipeline Execution

**Step-by-Step (Manual Control)**
```bash
# Execute notebooks in sequence:
# 1. notebooks/etl-1-fundacao-preparacao.ipynb
# 2. notebooks/etl-2-coleta-gitlab.ipynb  
# 3. notebooks/etl-3-processamento-docling.ipynb
# 4. notebooks/etl-4-segmentacao-chunks.ipynb
# 5. notebooks/etl-5-geracao-embeddings.ipynb
# 6. notebooks/etl-6-armazenamento-qdrant.ipynb
```

**Development and Testing**
```bash
# Each notebook can be run independently for development
# All validation and error handling is built into each stage
```

## Pipeline Execution Flow

Each notebook represents one stage of the pipeline:

1. **🏗️ Fundação**: Configuration loading, credential validation, environment setup
2. **📥 GitLab**: Repository connection, document download, file filtering
3. **⚙️ Docling**: Content extraction, OCR processing, structure preservation
4. **🔪 Chunking**: Text segmentation, overlap management, tokenization
5. **🧠 Embeddings**: Vector generation using BAAI/bge-m3 model
6. **💾 Qdrant**: Vector storage, collection management, indexing

## Data Passing Between Stages

### **Hybrid JSON + File System Approach**

Each stage outputs:
- **JSON Metadata** (`pipeline-data/metadata/stage_XX_name.json`) - Configuration, statistics, file lists
- **Actual Data Files** (`pipeline-data/stage_directory/`) - Documents, chunks, embeddings
- **Completion Lock** (`pipeline-data/checkpoints/stage_XX_completed.lock`) - Execution marker

Example data flow:
```json
// stage_02_gitlab.json
{
  "downloaded_files": [
    {
      "gitlab_path": "30-Aprovados/Manual.pdf",
      "local_path": "documents/Manual.pdf",
      "file_info": {"size_bytes": 2048576, "extension": ".pdf"}
    }
  ],
  "statistics": {"successfully_downloaded": 22, "total_size_mb": 156.7},
  "next_stage_instructions": {
    "documents_location": "documents/",
    "processing_order": "by_size_ascending"
  }
}
```

## Configuration System

Configuration is managed entirely within notebooks:

### Environment Files
```bash
.env.development    # Development settings
.env.staging       # Staging settings  
.env.production    # Production settings
```

### Key Configuration Areas
- **GitLab**: Repository URL, token, target folder
- **QDrant**: Vector database URL, API key, collection
- **Processing**: Chunk size, embedding model, batch sizes
- **Pipeline**: Concurrency, caching, monitoring

## Safety and Validation

### Dependency Checking
- Each notebook verifies previous stages completed
- Visual progress indicators show pipeline status
- Automatic prevention of out-of-order execution

### Error Handling
- Comprehensive error messages with troubleshooting guidance
- Production-ready validation and error recovery
- Detailed logging and error tracking

### Independent Execution
- Each notebook can be run independently for development
- Built-in dependency validation prevents execution errors
- Safe execution with comprehensive credential validation

## Service Integration

### GitLab Integration
- URL: Configurable (default: http://gitlab.processa.info)
- Repository: `nic/documentacao/base-de-conhecimento`
- Authentication: Token-based access
- Target Folder: `30-Aprovados`

### QDrant Vector Database
- URL: Configurable (default: https://qdrant.codrstudio.dev/)
- Collection: Environment-specific (`nic_dev`, `nic_prod`)
- Vector Size: 1024 dimensions (BAAI/bge-m3)
- Distance Metric: COSINE

### Docling Document Processing
- OCR: Automatic based on document type
- Languages: Portuguese and English
- Output: Structured JSON + clean text
- Confidence tracking and quality metrics

## Development Workflow

1. **Setup**: Configure `.env` file for your environment
2. **Development**: Use individual notebooks for iterative development
3. **Testing**: Run notebooks independently with real credentials for validation
4. **Integration**: Execute complete pipeline via master notebook
5. **Production**: Deploy with production environment configuration

## Troubleshooting

### Common Issues

1. **Dependency Errors**: Check notebook execution order with progress indicator
2. **Configuration Issues**: Validate `.env` settings in Foundation notebook
3. **GitLab Connection**: Verify token and repository access
4. **QDrant Issues**: Check API key and collection configuration
5. **Memory Issues**: Adjust batch sizes in notebook parameters

### Debug Tools

Each notebook includes debug functions:
```python
# Show detailed configuration
show_detailed_config()

# Show pipeline progress
show_pipeline_progress()

# Test connections
test_connections()

# Validate stage output
validate_stage_output()
```

### Data Inspection

All intermediate data is preserved and inspectable:
- `pipeline-data/metadata/` - Stage configurations and statistics
- `pipeline-data/documents/` - Downloaded files
- `pipeline-data/processed/` - Extracted content
- `pipeline-data/chunks/` - Text segments
- `pipeline-data/embeddings/` - Vector data

## Important Files

### Core Architecture
- `notebooks/etl.ipynb` - Main pipeline orchestrator
- `notebooks/rest-api.ipynb` - REST API endpoints (Jupyter Kernel Gateway)
- `src/proxy.py` - FastAPI reverse proxy for web dashboard
- `run-server.sh` - Main server startup script (proxy + kernel gateway)
- `jupyter_kernel_gateway_config.py` - Kernel Gateway configuration

### Pipeline Notebooks
- `notebooks/etl-1-fundacao-preparacao.ipynb` - Configuration and setup
- `notebooks/etl-2-coleta-gitlab.ipynb` - GitLab document collection
- `notebooks/etl-3-processamento-docling.ipynb` - Document processing
- `notebooks/etl-4-segmentacao-chunks.ipynb` - Text chunking
- `notebooks/etl-5-geracao-embeddings.ipynb` - Embedding generation
- `notebooks/etl-6-armazenamento-qdrant.ipynb` - Vector storage

### Web Dashboard
- `static/templates/base.html` - Jinja2 template with theme support
- `static/assets/css/styles.css` - Main stylesheet
- `static/assets/css/themes.css` - Theme system and semantic CSS classes
- `static/assets/js/app.js` - Frontend JavaScript with theme toggle
- `static/classes-semanticas.md` - Semantic CSS class documentation
- `static/index.md` - Homepage content

### Configuration
- `.env.*` - Environment-specific configuration
- `static/color-schema.md` - Color definitions for light/dark themes
- `pipeline-data/metadata/` - Runtime configurations

### Data Directories
- `pipeline-data/` - All pipeline data and state
- `logs/` - Execution logs and monitoring

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- **Web Framework**: `fastapi`, `uvicorn`, `jinja2`, `httpx`
- **Jupyter**: `jupyter`, `jupyter_kernel_gateway`
- **Document Processing**: `docling`, `pypdf`, `python-docx`, `markdown`
- **Machine Learning**: `transformers`, `torch`, `sentence-transformers`
- **Data Management**: `pandas`, `numpy`, `qdrant-client`
- **GitLab Integration**: `python-gitlab`
- **Utilities**: `tqdm`, `python-dotenv`, `pathlib`
- **Testing**: `pytest`, `pytest-cov`, `pytest-mock`, `pytest-asyncio`

## Docker Deployment

The project includes Docker deployment configuration:

```bash
# Build and run with Docker Compose
cd deployment/
docker-compose up -d

# Manual Docker build
docker build -t nic-etl .
docker run -v $(pwd)/pipeline-data:/app/pipeline-data nic-etl

# Deploy with monitoring (Prometheus + Grafana)
./deploy.sh --environment production --monitoring
```

See `deployment/DEPLOYMENT_GUIDE.md` for detailed deployment instructions.

## Testing

The project includes comprehensive testing capabilities:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=notebooks/src --cov-report=html

# Run specific test patterns
pytest -k "test_gitlab" -v

# Run tests in parallel
pytest -n auto
```

Testing framework includes:
- **Unit tests**: `pytest` with mocking capabilities
- **Coverage reporting**: Detailed test coverage analysis
- **Async testing**: Support for async/await patterns
- **HTTP mocking**: Mock external API calls with `responses`
- **Time-based testing**: Freeze time for deterministic tests

## Production Deployment

1. **Environment Setup**: Use `.env.production` with real credentials
2. **Dependency Installation**: Ensure all packages are installed
3. **Resource Planning**: Adequate disk space and memory
4. **Monitoring**: Enable logging and health checks
5. **Backup**: Backup QDrant collections before processing

The modular notebook architecture makes deployment straightforward - each stage can be monitored, debugged, and rerun independently as needed.

## Web Dashboard Architecture

The system provides a modern web interface built with FastAPI + Jupyter Kernel Gateway architecture:

### 🏗️ **Dual-Server Architecture**
- **FastAPI Proxy** (Port 5000) - Public interface serving static files and templates
- **Jupyter Kernel Gateway** (Port 5001) - Internal API server executing notebook endpoints
- **Reverse Proxy Pattern** - `/api/*` requests forwarded to Kernel Gateway seamlessly

### 🎨 **Frontend System**
- **Jinja2 Templates** - Server-side rendering with `base.html` template
- **Markdown Rendering** - Dynamic markdown-to-HTML conversion with syntax highlighting
- **Theme System** - Light/dark themes using CSS custom properties
- **Semantic CSS Classes** - Reusable components (`x-status`, `x-feature-card`, `x-alerta`, etc.)
- **Responsive Design** - Mobile-friendly with Ubuntu typography

### 🔄 **API Integration**
- **REST API Notebook** - `notebooks/rest-api.ipynb` defines all endpoints
- **Background Jobs** - Pipeline execution independent of HTTP connections
- **Status Monitoring** - Real-time pipeline status via JSON APIs
- **Error Handling** - Comprehensive error responses and logging

### 📝 **Content Management**
- **Fallback System** - HTML files take precedence over Markdown files
- **Live Markdown** - Pages automatically rendered from `.md` files in `static/`
- **Asset Pipeline** - CSS, JavaScript, and images served from `static/assets/`

## CSS Architecture

### 🎨 **Theme System**
Uses CSS custom properties for seamless light/dark theme switching:

```css
:root {
    --color-blue-light: #3d95df;    /* Light theme blue */
    --color-blue-dark: #5fbcd3;     /* Dark theme blue */
    --accent-primary: var(--color-blue);
}

[data-theme="light"] { --color-blue: var(--color-blue-light); }
[data-theme="dark"] { --color-blue: var(--color-blue-dark); }
```

### 🧱 **Semantic Class System**
Instead of inline styles, use semantic CSS classes defined in `static/assets/css/themes.css`:

- **Status**: `.x-status`, `.x-status-loading`
- **Layout**: `.x-features-grid`, `.x-feature-card`  
- **Messages**: `.x-alerta`, `.x-warning`, `.x-success`, `.x-tip`, `.x-note`
- **Content**: `.x-info`, `.x-mission`, `.x-example`, `.x-config`
- **Pipeline**: `.x-pipeline-stage`

Each class automatically adapts to theme changes and includes appropriate icons via CSS `::before` pseudo-elements.

## Development Guidelines

### 🚫 **Important Rules**
- **Never modify notebook files** unless explicitly required - they are the core pipeline
- **Use semantic CSS classes** instead of inline styles in markdown content
- **Test theme switching** when modifying CSS or adding new components
- **Follow the dual-server architecture** - proxy serves UI, Kernel Gateway handles API

### ✅ **Best Practices**
- Use `./run-server.sh` for development - it handles both servers correctly
- Check port availability (5000/5001) before starting services
- Use semantic classes from `static/classes-semanticas.md` for consistent styling
- Test both light and dark themes when adding UI components
- Monitor logs in `logs/` directory for debugging server issues