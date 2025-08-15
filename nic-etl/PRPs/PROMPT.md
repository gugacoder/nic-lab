## **OBJETIVO**

implemente um junyper notebook chamado NIC ETL que se conecta a uma pasta de um branc de um repositorio de um gitlab especifico e realiza o processamento de embeddings inserindo os dados em um qdrant.

**notas:**
- se o catalogo no qdrant nao existir, o notebook deve cria-lo

## **PARAMETROS:**

GITLAB
url: http://gitlab.processa.info/nic/documentacao/base-de-conhecimento.git
ACCESS TOKEN: glpat-zycwWRydKE53SHxxpfbN
branch: main
folder: 30-Aprovados

QDRANT
url: https://qdrant.codrstudio.dev/
api_key: 93f0c9d6b9a53758f2376decf318b3ae300e9bdb50be2d0e9c893ee4469fd857
catalog: nic

## **PIPELINE**

1. **ingestão de documentos**

   1. coletar arquivos-alvo (GitLab)
   2. normalizar formatos de entrada (PDF/DOCX/IMG)

2. **pré-processamento & OCR**

   1. detectar tipo de conteúdo (texto digital × imagem/escaneado)
   2. aplicar **OCR open source (Tesseract + OCRmyPDF + Poppler)** quando escaneado
   3. gerar saída pesquisável unificada por documento (texto + layout + page map)

3. **structuring (docling)**

   1. analisar estrutura lógica (títulos, seções, parágrafos)
   2. extrair o texto estruturado para downstream

4. **chunker**

   1. chunk strategy: **paragraph**
   2. chunk size: **500 tokens**
   3. chunk overlap: **100 tokens**
   4. **token counting**: usar **tokenizer do BGE** para medir 500/100

5. **metadata extractor (NIC Schema)**

   1. adicionar **document metadata** aos chunks
   2. adicionar **section metadata** aos chunks
   3. registrar lineage/processamento (ex.: `ocr=true/false`, repo/commit, is\_latest)

6. **embedder**

   1. embedding model: **BAAI/bge-m3 (local, CPU)**
   2. dimensions: **1024**

7. **inserção no Qdrant (database `nic`)**

   1. criar a coleção se não existir (**size=1024**, distância COSINE)
   2. upsert de chunks com payload (NIC Schema) e IDs estáveis


### NIC Schema Metadata

```json
{
  "document": {
    "type": "object",
    "properties": {
      "related": {
        "type": "array",
        "items": {
          "type": "string"
        }
      },
      "author": {
        "type": "array",
        "items": {
          "type": "string"
        }
      },
      "created": {
        "format": "date",
        "type": "string"
      },
      "description": {
        "type": "string"
      },
      "up": {
        "type": "array",
        "items": {
          "type": "string"
        }
      },
      "title": {
        "type": "string"
      },
      "status": {
        "type": "string",
        "enum": [
          "rascunho",
          "revisão",
          "publicado",
          "arquivado"
        ]
      },
      "tags": {
        "type": "array",
        "items": {
          "type": "string"
        }
      }
    },
    "required": [
      "title",
      "description",
      "status",
      "created"
    ]
  },
  "sections": {}
}
```