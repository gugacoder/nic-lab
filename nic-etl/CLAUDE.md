# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NIC ETL is a comprehensive ETL pipeline for the NIC (Núcleo de Inteligência e Conhecimento) that processes documents from GitLab repositories, creates high-quality embeddings, and stores processed data in QDrant for efficient semantic search. The project is built around a Jupyter Notebook-based architecture with modular Python components.

## Architecture

The system follows a **modular architecture** with clear separation of concerns:

- **Main Pipeline Notebook** (`nic_etl_pipeline.ipynb`): Central orchestrator that coordinates all pipeline stages
- **Configuration System** (`test_config.py`): Centralized configuration management with environment-specific settings
- **Modular Components**: Each pipeline stage is implemented as a separate, testable module
- **Error Handling & Monitoring**: Comprehensive error handling with retry policies and health monitoring

### Key Components

1. **Configuration Management**: Environment-aware configuration system supporting development/staging/production
2. **GitLab Integration**: Document extraction from specific GitLab repositories and folders
3. **Document Processing**: Docling-based document structuring with OCR capabilities
4. **Text Chunking**: Token-based chunking using BAAI/bge-m3 tokenizer (500 tokens, 100 overlap)
5. **Embedding Generation**: CPU-based embeddings using BAAI/bge-m3 model (1024 dimensions)
6. **Vector Storage**: QDrant integration with NIC Schema metadata

## Development Commands

### Running the Pipeline
```bash
# Start Jupyter Lab
jupyter lab

# Execute the main pipeline notebook
# Open: nic_etl_pipeline.ipynb
```

### Configuration Testing
```bash
# Test configuration system
python test_config.py
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies (as defined in notebook dependency manager)
# Dependencies are automatically managed by the notebook's DependencyManager
```

### Testing and Validation
```bash
# Run tests (when implemented)
python -m pytest tests/

# Linting (when configured)
flake8 src/
black src/
```

## Configuration System

The project uses a sophisticated configuration management system (`test_config.py`) that:

- Supports multiple environments (development, staging, production)
- Loads settings from `.env` files with fallback to defaults
- Provides environment-specific overrides
- Validates configuration integrity
- Manages feature flags and performance settings

### Key Configuration Sections

- **GitLab**: Repository connection and authentication
- **QDrant**: Vector database connection and collection settings
- **Docling**: OCR and document processing configuration
- **Chunking**: Text segmentation parameters
- **Embedding**: Model and processing settings
- **Cache**: Caching and state management
- **Performance**: Worker threads, timeouts, and batch sizes

### Environment Variables

Copy configuration template and adjust for your environment:
```bash
cp .env.example .env  # When available
# Edit .env with your specific values
```

## Pipeline Execution Flow

1. **Environment Setup**: Configuration loading and dependency validation
2. **GitLab Connection**: Authenticate and connect to repository
3. **Document Retrieval**: Extract documents from specified folder/branch
4. **Document Processing**: Docling-based structuring with conditional OCR
5. **Text Chunking**: Segment content into overlapping chunks
6. **Embedding Generation**: Create vectors using BAAI/bge-m3
7. **QDrant Storage**: Insert chunks with NIC Schema metadata

## Error Handling and Monitoring

The system includes comprehensive error handling with:

- **Retry Policies**: Configurable retry logic with exponential backoff
- **Error Categorization**: Network, processing, validation, resource errors
- **Health Monitoring**: System health checks and performance metrics
- **Alerting**: Critical error notifications and logging

## Key Design Principles

1. **Modularity**: Each component is independently testable and reusable
2. **Idempotency**: Pipeline reruns don't create duplicates
3. **Error Resilience**: Graceful handling of partial failures
4. **Configuration-Driven**: Environment-aware settings management
5. **Production-Ready**: Designed for development, staging, and production use

## Important Files

- `nic_etl_pipeline.ipynb`: Main pipeline orchestrator and interface
- `test_config.py`: Configuration management system and validation
- `PRPs/PROMPT.md`: Detailed implementation requirements and specifications
- `cache/`: Pipeline state and document caching
- `logs/`: Execution logs and error tracking

## Service Integration

### GitLab
- URL: Configurable (default: http://gitlab.processa.info)
- Authentication: Token-based access
- Target: `nic/documentacao/base-de-conhecimento` repository
- Branch: `main`, Folder: `30-Aprovados`

### QDrant Vector Database
- URL: Configurable (default: https://qdrant.codrstudio.dev/)
- Collection: `nic`
- Vector Size: 1024 dimensions
- Distance Metric: COSINE

### Docling Document Processing
- OCR: Conditional based on document type
- Languages: Portuguese and English
- Confidence Threshold: 75%
- Output: Structured JSON/Markdown with metadata

## Development Workflow

1. **Configuration**: Set up environment variables and validate configuration
2. **Development**: Use Jupyter Lab for interactive development
3. **Testing**: Run configuration tests and pipeline validation
4. **Production**: Deploy with production environment settings

## Dependencies

The system automatically manages dependencies through the notebook's `DependencyManager`, including:

- Core processing: `docling`, `transformers`, `torch`, `sentence-transformers`
- Data handling: `pandas`, `numpy`, `python-gitlab`
- Vector database: `qdrant-client`
- Document processing: `pypdf`, `python-docx`, `pillow`
- Utilities: `tqdm`, `psutil`, `python-dotenv`

## Troubleshooting

### Common Issues

1. **Configuration Errors**: Run `python test_config.py` to validate settings
2. **Dependency Issues**: The notebook's DependencyManager handles installation
3. **GitLab Connection**: Verify token and repository access
4. **QDrant Issues**: Check API key and collection configuration
5. **Memory Issues**: Adjust batch sizes and worker counts in configuration

### Logs and Monitoring

- Execution logs: `./logs/nic_etl.log`
- Error alerts: `./logs/alerts.log`
- Pipeline state: `./cache/pipeline_state.json`

Use the built-in health monitoring system to check system status and performance metrics.