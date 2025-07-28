
# QDrant Data Quality Testing Application

Uma aplicação Flask completa para testar qualidade de dados no Qdrant vector database.

## Características

- ✅ Interface web integrada com layout responsivo
- ✅ Conecta ao Qdrant via API REST
- ✅ Suporte a múltiplos algoritmos de embedding (sentence-transformers)
- ✅ Busca vetorial com resultados formatados em markdown
- ✅ Persistência de configurações entre sessões
- ✅ Suporte HTTPS com certificado auto-assinado
- ✅ Tratamento robusto de erros
- ✅ Arquivo Python único executável

## Instalação

1. Instale as dependências:
```bash
pip install flask sentence-transformers requests
```

Ou usando o arquivo requirements.txt:
```bash
pip install -r requirements.txt
```

2. Execute a aplicação:
```bash
python qdrant_tester.py
```

3. Acesse no navegador:
- HTTP: http://localhost:5000
- HTTPS: https://localhost:5000 (certificado auto-assinado)

## Como Usar

### 1. Configuração Inicial
- **Host**: URL do seu servidor Qdrant (ex: https://seu-qdrant:6333)
- **API Key**: Chave de API opcional para autenticação
- **Embedding Algorithm**: Algoritmo para gerar embeddings (padrão: all-MiniLM-L6-v2)
- **Vector Field Name**: Nome do campo vetor na collection (padrão: "vector")
- **Max Results**: Número máximo de resultados (padrão: 10)

### 2. Buscar Collections
1. Preencha Host e API Key (se necessário)
2. Clique em "Buscar Collections"
3. Selecione uma collection do dropdown

### 3. Executar Consulta
1. Digite sua consulta no campo "Query Input"
2. Clique em "Submit Query"
3. Veja os resultados formatados em markdown

## Formato dos Resultados

```markdown
# Query: sua consulta aqui

Found X results:

---
#1 (Score: 0.8532)
Match: texto encontrado no documento...
Metadata: {
    "id": "doc_123",
    "category": "exemplo",
    "timestamp": "2024-01-15"
}
---
#2 (Score: 0.7891)
Match: outro texto encontrado...
Metadata: {
    "id": "doc_456",
    "tags": ["tag1", "tag2"]
}
---
```

## Algoritmos de Embedding Suportados

- **all-MiniLM-L6-v2** (384 dimensões) - Padrão, rápido e eficiente
- **all-mpnet-base-v2** (768 dimensões) - Maior qualidade, mais lento
- **all-MiniLM-L12-v2** (384 dimensões) - Balanceado

## Persistência

A aplicação salva automaticamente as configurações em:
- **Cliente**: localStorage do navegador
- **Servidor**: arquivo `qdrant_config.json`

## Tratamento de Erros

A aplicação trata os seguintes cenários:
- Conexão com Qdrant falhou
- API Key inválida
- Collection não encontrada
- Timeout de requisições
- Modelos de embedding não disponíveis
- Formato de resposta inválido

## Tecnologias Utilizadas

- **Flask**: Framework web Python
- **sentence-transformers**: Geração de embeddings
- **requests**: Cliente HTTP para API REST
- **HTML/CSS/JavaScript**: Interface web responsiva

## Arquitetura

```
qdrant_tester.py
├── Flask App (Backend)
├── HTML Template (Frontend)
├── QdrantTester Class
│   ├── fetch_collections()
│   ├── search_vectors()
│   ├── get_embedding_model()
│   └── format_results_markdown()
└── API Endpoints
    ├── /api/collections
    └── /api/search
```

## Licença

Código livre para uso pessoal e comercial.
