# ROCIN

## **Role**

Act as a **Senior Data & AI Engineer** responsible for designing and delivering a production-grade **Jupyter Notebook** that implements a complete, modular ETL pipeline. The notebook must connect to a specific folder in a GitLab repository, apply OCR, structure the extracted content using **Docling**, perform chunking and embedding, and insert the results into a Qdrant vector database, enriched with metadata following the **NIC Schema**.

---

## **Objective**

Create a Jupyter Notebook named **NIC ETL** that:

* Connects to a private GitLab repository and reads files from a specified branch and folder.
* Normalizes file formats and applies OCR when required.
* Structures the textual content using **Docling**.
* Chunks the structured content into token-based segments.
* Generates embeddings using the `BAAI/bge-m3` model (CPU-based).
* Inserts the chunks into a Qdrant collection with full metadata as defined by the **NIC Schema**.

---

## **Context**

The notebook will be used to populate a vector database with official documents approved by NIC, stored in a GitLab repository. Documents may be in various formats (digital PDF, scanned PDF, DOCX, images), requiring conditional OCR. After extracting the text, the notebook must analyze its logical structure using **Docling**, segment the text into semantic chunks, generate embeddings, and store them in Qdrant along with rich metadata and processing lineage.

---

## **Implementation Structure**

**CRITICAL REQUIREMENT**: All implementation must be contained within Jupyter Notebook cells only.

* **DO NOT** create separate .py files, src/ directory structure, or external Python modules
* **DO NOT** create traditional Python package architecture with __init__.py files
* **"Modular"** means "organized in separate notebook cells", NOT "separate Python files"
* Each major processing stage should be implemented in its own notebook cell
* Helper functions should be defined in the same cell where they're used
* All code, classes, and functions must remain within the notebook

**Expected Notebook Cell Structure:**
- **Cell 1**: Dependencies Installation and Configuration
- **Cell 2**: GitLab Connection and Authentication Functions
- **Cell 3**: Document Retrieval and Caching Functions
- **Cell 4**: Docling Processing and OCR Functions
- **Cell 5**: Text Chunking Functions
- **Cell 6**: Embedding Generation Functions
- **Cell 7**: Qdrant Integration Functions
- **Cell 8**: Metadata Management (NIC Schema)
- **Cell 9**: Main Pipeline Orchestration
- **Cell 10**: Testing and Validation Functions
- **Cell 11**: Pipeline Execution and Monitoring

---

## **Instructions**

### 1. Ingestion

* Access the following GitLab repository:

  * URL: `http://gitlab.processa.info/nic/documentacao/base-de-conhecimento.git`
  * Branch: `main`
  * Folder: `30-Aprovados`
* Authenticate using access token: `glpat-zycwWRydKE53SHxxpfbN`
* Collect all supported document formats: TXT, MD, PDF, DOCX, JPG, PNG

### 2. Unified Conversion

* Ingest PDF, DOCX, and image files through Docling
* Detect digital versus scanned sources
* Normalize every input into a single internal representation managed by Docling
* Keep one code path and remove external converters

### 3. OCR and Text Extraction

* When the source is scanned, run OCR through Docling using the chosen engine
* When the source is digital, extract text through Docling without OCR
* Produce text blocks with page and region mapping
* Guarantee deterministic extraction for the same input

### 4. Document Structuring and Delivery

* Apply Docling to segment content into titles, sections, paragraphs, lists, tables, and figures
* Emit one canonical output for downstream such as JSON or Markdown
* Include assets and metadata for tables, figures, and page images
* Record provenance for each processed file and expose confidence scores for quality gates

### 5. Chunking

* Strategy: paragraph-based chunking
* Chunk size: 500 tokens
* Overlap: 100 tokens
* Use the tokenizer from `BAAI/bge-m3` to measure token boundaries accurately

### 6. Metadata Enrichment (NIC Schema)

* For each chunk, attach:

  * Document metadata: `title`, `description`, `status`, `created`, etc.
  * Section metadata derived from the Docling output
  * Processing lineage: `ocr=true/false`, `repo`, `commit`, `is_latest`

### 7. Embedding

* Model: `BAAI/bge-m3`
* Run locally on CPU
* Embedding size: 1024 dimensions

### 8. Qdrant Insertion

* Qdrant settings:

  * URL: `https://qdrant.codrstudio.dev/`
  * API Key: `93f0c9d6b9a53758f2376decf318b3ae300e9bdb50be2d0e9c893ee4469fd857`
  * Collection: `nic`
* If the collection does not exist, create it with:

  * Vector size: 1024
  * Distance: COSINE
* Insert each chunk with:

  * Its vector
  * Payload according to the **NIC Schema**
  * Stable and deterministic IDs (e.g., UUID5 or content hash)

---

## **Notes**

* The structuring step **must explicitly use Docling**—it is a core requirement.
* **JUPYTER NOTEBOOK ONLY**: All code must be implemented within notebook cells - no external .py files
* **"Modular sections"** refers to organized notebook cells, not separate Python modules
* **"Reusable sections"** means functions within cells that can be called multiple times
* Include installation and configuration of all required dependencies **in the first notebook cell**
* The pipeline must be idempotent—reruns should not create duplicates in Qdrant.
* Handle partial failures (e.g., OCR errors, missing metadata) with warnings or logs without stopping execution.
* Validate payloads against the provided **NIC Schema**.
* Each processing stage should be clearly documented within its respective cell
* Use markdown cells to separate and document each major section of the pipeline
