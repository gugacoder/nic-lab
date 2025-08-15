# NIC ETL - Núcleo de Inteligência e Conhecimento

Este projeto implementa o pipeline ETL (Extract, Transform, Load) para o NIC (Núcleo de Inteligência e Conhecimento), a solução de Inteligência Artificial da Processa Sistemas.

## 📋 Visão Geral

O NIC ETL é responsável por processar documentos do GitLab, criar embeddings de alta qualidade e armazenar os dados processados no QDrant para consultas eficientes de busca semântica. O projeto utiliza Jupyter Notebooks para facilitar a experimentação e o desenvolvimento interativo do pipeline de dados.

## 🎯 Objetivos

- **Extração**: Coleta de documentos e dados do GitLab
- **Transformação**: Processamento e criação de embeddings vetoriais de qualidade
- **Carregamento**: Injeção dos dados processados no banco vetorial QDrant

## 🏗️ Arquitetura

```
GitLab → ETL Pipeline → QDrant Vector DB
   ↓         ↓              ↓
Documentos → Embeddings → Busca Semântica
```

## 🚀 Começando

### Pré-requisitos

- Python 3.8+
- Jupyter Notebook ou JupyterLab
- Acesso ao GitLab da Processa Sistemas
- Instância do QDrant (local ou na nuvem)

### Instalação

1. Clone o repositório:
```bash
git clone <repository-url>
cd nic-etl
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Configuração

1. Configure as variáveis de ambiente:
```bash
cp .env.example .env
```

2. Edite o arquivo `.env` com suas credenciais:
```
GITLAB_URL=https://gitlab.processa.com
GITLAB_TOKEN=seu_token_aqui
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=sua_chave_aqui
```

## 📊 Notebooks Disponíveis

### 1. Data Extraction (`01_extraction.ipynb`)
- Conexão com a API do GitLab
- Extração de documentos, issues e merge requests
- Filtros e seleção de dados relevantes

### 2. Data Processing (`02_processing.ipynb`)
- Limpeza e normalização de dados
- Chunking de documentos longos
- Pré-processamento de texto

### 3. Embedding Generation (`03_embeddings.ipynb`)
- Geração de embeddings usando modelos state-of-the-art
- Otimização de qualidade e performance
- Validação de embeddings

### 4. Data Loading (`04_loading.ipynb`)
- Configuração de coleções no QDrant
- Upload de embeddings e metadados
- Indexação e otimização

## 🔧 Configuração do QDrant

### Instalação Local
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Configuração de Coleção
```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="nic_documents",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
```

## 🔍 Uso

1. Inicie o Jupyter:
```bash
jupyter lab
```

2. Execute os notebooks em sequência:
   - `01_extraction.ipynb`
   - `02_processing.ipynb`
   - `03_embeddings.ipynb`
   - `04_loading.ipynb`

3. Monitore o progresso através dos logs e métricas exibidos

## 📈 Monitoramento e Métricas

- **Documentos processados**: Contador de documentos extraídos
- **Qualidade dos embeddings**: Métricas de similaridade e coerência
- **Performance**: Tempo de processamento e throughput
- **Erros**: Log de falhas e recuperação

## 🛠️ Desenvolvimento

### Estrutura do Projeto
```
nic-etl/
├── notebooks/          # Jupyter notebooks
├── src/               # Código fonte Python
├── config/            # Arquivos de configuração
├── data/              # Dados temporários e cache
├── logs/              # Logs de execução
├── tests/             # Testes unitários
└── requirements.txt   # Dependências
```

### Executando Testes
```bash
python -m pytest tests/
```

### Linting
```bash
flake8 src/
black src/
```

## 🔐 Segurança

- Nunca commite credenciais no repositório
- Use variáveis de ambiente para informações sensíveis
- Mantenha tokens e chaves de API seguros
- Implemente autenticação adequada para acesso ao QDrant

## 📝 Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é propriedade da Processa Sistemas. Todos os direitos reservados.

## 🤝 Suporte

Para dúvidas e suporte:
- Email: nic@processa.com.br
- Slack: #nic-support
- Documentação: [Wiki Interno](link-para-wiki)

## 🚀 Roadmap

- [ ] Integração com outros repositórios Git
- [ ] Suporte a múltiplos modelos de embedding
- [ ] Interface web para monitoramento
- [ ] Pipeline automatizado com CI/CD
- [ ] Métricas avançadas de qualidade
- [ ] Suporte a diferentes formatos de documento

---

**NIC - Núcleo de Inteligência e Conhecimento**  
*Processa Sistemas - Transformando dados em conhecimento*