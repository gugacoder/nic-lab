# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NIC ETL is a comprehensive ETL pipeline for the NIC (Núcleo de Inteligência e Conhecimento) that processes documents from GitLab repositories, creates high-quality embeddings, and stores processed data in QDrant for efficient semantic search. The project is built around a **modular Jupyter Notebook architecture** where each notebook represents a pipeline stage.

## Modular Architecture

The system follows a **notebook-centric modular architecture** where the pipeline IS the notebooks:

### 🌳 **Pipeline Structure (7 Notebooks)**
```
📚 notebooks/
├── 🚀 00_PIPELINE_MASTER.ipynb         # Orquestrador principal
├── 🏗️ 01_FUNDACAO_PREPARACAO.ipynb     # Configuração e validação  
├── 📥 02_COLETA_GITLAB.ipynb           # Download de documentos
├── ⚙️ 03_PROCESSAMENTO_DOCLING.ipynb   # Extração de conteúdo
├── 🔪 04_SEGMENTACAO_CHUNKS.ipynb      # Segmentação de texto
├── 🧠 05_GERACAO_EMBEDDINGS.ipynb      # Geração de vetores
├── 💾 06_ARMAZENAMENTO_QDRANT.ipynb    # Inserção vetorial
└── 📊 07_VALIDACAO_RESULTADOS.ipynb    # Testes e métricas
```

### 🐍 **Supporting Python Library** (`src/`)
- **Ultra-simplified functions** that trust 100% in notebook parameters
- **No internal validations** - notebooks handle all validation logic
- **Pure algorithms** - each function does one thing well
- **Notebook-driven design** - Python serves the notebooks, not the other way around

### 📦 **Data Flow Architecture**
```
📁 pipeline_data/
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

### Running the Pipeline

**Option 1: Complete Pipeline (Automated)**
```bash
# Start Jupyter Lab
jupyter lab

# Open and execute the master notebook
# File: notebooks/00_PIPELINE_MASTER.ipynb
# Run all cells for full automation
```

**Option 2: Step-by-Step (Manual Control)**
```bash
# Execute notebooks in sequence:
# 1. notebooks/01_FUNDACAO_PREPARACAO.ipynb
# 2. notebooks/02_COLETA_GITLAB.ipynb  
# 3. notebooks/03_PROCESSAMENTO_DOCLING.ipynb
# 4. notebooks/04_SEGMENTACAO_CHUNKS.ipynb
# 5. notebooks/05_GERACAO_EMBEDDINGS.ipynb
# 6. notebooks/06_ARMAZENAMENTO_QDRANT.ipynb
# 7. notebooks/07_VALIDACAO_RESULTADOS.ipynb
```

**Option 3: Development and Testing**
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
7. **📊 Validation**: Quality tests, search validation, performance metrics

## Data Passing Between Stages

### **Hybrid JSON + File System Approach**

Each stage outputs:
- **JSON Metadata** (`pipeline_data/metadata/stage_XX_name.json`) - Configuration, statistics, file lists
- **Actual Data Files** (`pipeline_data/stage_directory/`) - Documents, chunks, embeddings
- **Completion Lock** (`pipeline_data/checkpoints/stage_XX_completed.lock`) - Execution marker

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
- `pipeline_data/metadata/` - Stage configurations and statistics
- `pipeline_data/documents/` - Downloaded files
- `pipeline_data/processed/` - Extracted content
- `pipeline_data/chunks/` - Text segments
- `pipeline_data/embeddings/` - Vector data

## Important Files

### Core Notebooks
- `notebooks/00_PIPELINE_MASTER.ipynb` - Main orchestrator
- `notebooks/01_FUNDACAO_PREPARACAO.ipynb` - Configuration and setup
- `notebooks/0X_*.ipynb` - Individual pipeline stages

### Configuration
- `.env.*` - Environment-specific configuration
- `pipeline_data/metadata/` - Runtime configurations

### Documentation  
- `PRPs/PROMPT.md` - Original implementation requirements
- `README.md` - Project overview and setup
- This file - Development guidance

### Data Directories
- `pipeline_data/` - All pipeline data and state
- `logs/` - Execution logs and monitoring

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- **Document Processing**: `docling`, `pypdf`, `python-docx`
- **Machine Learning**: `transformers`, `torch`, `sentence-transformers`
- **Data Management**: `pandas`, `numpy`, `qdrant-client`
- **GitLab Integration**: `python-gitlab`
- **Utilities**: `tqdm`, `python-dotenv`, `pathlib`

## Production Deployment

1. **Environment Setup**: Use `.env.production` with real credentials
2. **Dependency Installation**: Ensure all packages are installed
3. **Resource Planning**: Adequate disk space and memory
4. **Monitoring**: Enable logging and health checks
5. **Backup**: Backup QDrant collections before processing

The modular notebook architecture makes deployment straightforward - each stage can be monitored, debugged, and rerun independently as needed.