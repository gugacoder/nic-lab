# 🚀 NIC ETL Pipeline - Sistema Modular de Notebooks

Sistema modular de ETL para o NIC (Núcleo de Inteligência e Conhecimento) que processa documentos do GitLab, extrai conteúdo estruturado e cria embeddings vetoriais para busca semântica.

## ✨ Características Principais

- 📓 **100% Jupyter Notebooks** - Pipeline transparente e autodocumentado
- 🔗 **Execução Modular** - 7 notebooks independentes e interconectados
- 🛡️ **Execução Segura** - Validação de dependências e ordem de execução
- 🧪 **Modo Teste** - Cada notebook pode ser executado independentemente
- 📊 **Dados Transparentes** - Fluxo de dados completamente visível (JSON + arquivos)

## 🌳 Estrutura do Pipeline

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

## 📦 Fluxo de Dados

```
GitLab → Documents → Docling → Chunks → Embeddings → Qdrant
   ↓        ↓          ↓        ↓         ↓         ↓
📁docs   📄JSON    📝text   ✂️pieces   🧠vectors  🔍search
```

## 🚀 Início Rápido

### 1. Configuração do Ambiente

```bash
# Clone o repositório
git clone <repository-url>
cd nic-etl

# Instale dependências
pip install -r requirements.txt

# Configure variáveis de ambiente
cp .env.example .env.development
# Edite .env.development com suas credenciais
```

### 2. Configuração das Credenciais

Edite `.env.development`:
```bash
# GitLab
GITLAB_URL=http://gitlab.processa.info
GITLAB_ACCESS_TOKEN=seu_token_aqui
GITLAB_REPOSITORY=nic/documentacao/base-de-conhecimento
GITLAB_TARGET_FOLDER=30-Aprovados

# Qdrant
QDRANT_URL=https://qdrant.codrstudio.dev/
QDRANT_API_KEY=sua_api_key_aqui
QDRANT_COLLECTION=nic_dev

# Processamento
CHUNK_SIZE=500
CHUNK_OVERLAP=100
EMBEDDING_MODEL=BAAI/bge-m3
```

### 3. Execução do Pipeline

**Opção A: Pipeline Completo (Automático)**
```bash
# Inicie Jupyter Lab
jupyter lab

# Abra e execute: notebooks/00_PIPELINE_MASTER.ipynb
# Execute todas as células para automação completa
```

**Opção B: Execução Manual (Passo a Passo)**
```bash
# Execute os notebooks em sequência:
# 1. notebooks/01_FUNDACAO_PREPARACAO.ipynb
# 2. notebooks/02_COLETA_GITLAB.ipynb  
# 3. notebooks/03_PROCESSAMENTO_DOCLING.ipynb
# 4. notebooks/04_SEGMENTACAO_CHUNKS.ipynb
# 5. notebooks/05_GERACAO_EMBEDDINGS.ipynb
# 6. notebooks/06_ARMAZENAMENTO_QDRANT.ipynb
# 7. notebooks/07_VALIDACAO_RESULTADOS.ipynb
```

**Opção C: Teste Independente**
```python
# Em qualquer notebook, configure:
MODO_INDEPENDENTE = True
# Execute o notebook para teste isolado
```

## 📋 Funcionalidades de Cada Etapa

### 🏗️ **Etapa 1: Fundação e Preparação**
- Carregamento e validação de configurações
- Teste de conectividade com GitLab e Qdrant
- Preparação do ambiente e diretórios
- Validação de credenciais

### 📥 **Etapa 2: Coleta GitLab**
- Conexão com repositório GitLab
- Download de documentos da pasta especificada
- Filtragem por tipo e tamanho de arquivo
- Coleta de metadados dos documentos

### ⚙️ **Etapa 3: Processamento Docling**
- Extração de conteúdo com Docling
- OCR automático quando necessário
- Preservação de estrutura (títulos, seções)
- Geração de texto limpo

### 🔪 **Etapa 4: Segmentação em Chunks**
- Divisão inteligente em pedaços menores
- Preservação de contexto com overlap
- Respeito a limites de frases
- Tokenização adequada para embeddings

### 🧠 **Etapa 5: Geração de Embeddings**
- Criação de vetores com BAAI/bge-m3
- Processamento em lotes otimizados
- Normalização e validação de qualidade
- Preparação para armazenamento vetorial

### 💾 **Etapa 6: Armazenamento Qdrant**
- Inserção de vetores no Qdrant
- Gerenciamento de collections
- Indexação otimizada
- Controle de duplicatas

### 📊 **Etapa 7: Validação e Resultados**
- Testes de busca semântica
- Validação de qualidade dos embeddings
- Métricas de performance
- Relatórios finais

## 🔧 Configuração Avançada

### Ambientes Múltiplos

```bash
.env.development    # Desenvolvimento
.env.staging       # Teste/Homologação
.env.production    # Produção
```

### Parâmetros Principais

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `CHUNK_SIZE` | Tamanho dos chunks em tokens | 500 |
| `CHUNK_OVERLAP` | Sobreposição entre chunks | 100 |
| `EMBEDDING_MODEL` | Modelo para embeddings | BAAI/bge-m3 |
| `BATCH_SIZE` | Tamanho do lote de processamento | 32 |
| `MAX_CONCURRENT_DOCS` | Documentos simultâneos | 4 |

## 🛡️ Segurança e Validação

### Sistema de Dependências
- Cada notebook verifica se etapas anteriores foram concluídas
- Validação automática de ordem de execução
- Indicadores visuais de progresso

### Modo de Teste
- Cada notebook pode executar independentemente
- Dados mock para desenvolvimento
- Testes sem afetar dados de produção

### Monitoramento
- Logs detalhados de cada etapa
- Checkpoints de conclusão
- Relatórios de qualidade

## 📊 Dados Gerados

### Estrutura de Dados
```
📁 pipeline-data/
├── 🗂️ documents/      # Documentos baixados
├── 🧾 processed/      # Conteúdo extraído
├── ✂️ chunks/         # Segmentos de texto
├── 🧠 embeddings/     # Dados vetoriais
├── 📊 metadata/       # Configurações e estatísticas
└── 🔄 checkpoints/    # Marcadores de conclusão
```

### Formatos de Saída
- **JSON**: Metadados, configurações, estatísticas
- **TXT**: Texto puro extraído
- **Arquivos binários**: Embeddings e vetores
- **Logs**: Histórico de execução

## 🔍 Troubleshooting

### Problemas Comuns

1. **Erro de Dependências**
   - Verifique ordem de execução dos notebooks
   - Use `show_pipeline_progress()` para ver status

2. **Problemas de Configuração**
   - Valide credenciais no notebook 01_FUNDACAO
   - Verifique arquivo `.env`

3. **Falhas de Conexão**
   - Teste conectividade GitLab/Qdrant
   - Verifique tokens e URLs

4. **Problemas de Memória**
   - Ajuste `BATCH_SIZE` e `MAX_CONCURRENT_DOCS`
   - Monitore uso de recursos

### Ferramentas de Debug

Cada notebook inclui funções de debug:
```python
show_pipeline_progress()    # Progresso atual
show_detailed_config()     # Configuração completa
test_connections()          # Teste de conectividade
validate_stage_output()     # Validação de saída
```

## 🏭 Produção

### Checklist de Deploy

- [ ] Configurar `.env.production` com credenciais reais
- [ ] Instalar todas as dependências
- [ ] Verificar recursos (CPU, memória, disco)
- [ ] Configurar monitoramento e logs
- [ ] Fazer backup das collections Qdrant
- [ ] Testar em staging primeiro

### Monitoramento

```python
# Verificar saúde do sistema
check_system_health()

# Estatísticas de execução
show_execution_stats()

# Validar qualidade dos embeddings
validate_embedding_quality()
```

## 🤝 Contribuição

1. Fork o repositório
2. Crie uma branch para sua feature
3. Teste em modo independente
4. Envie pull request com documentação

## 📄 Licença

Este projeto é licenciado sob [LICENSE] - veja o arquivo LICENSE para detalhes.

## 🆘 Suporte

- 📖 **Documentação**: Consulte `CLAUDE.md` para detalhes técnicos
- 🐛 **Issues**: Reporte problemas no GitHub
- 💬 **Discussões**: Use Discussions para dúvidas

---

**🎯 Pipeline NIC ETL - Transformando documentos em conhecimento pesquisável**