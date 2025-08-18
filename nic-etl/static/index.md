# 🚀 NIC ETL Pipeline

Bem-vindo ao **Núcleo de Inteligência e Conhecimento** - um sistema completo de ETL para processamento inteligente de documentos.

## ⚡ Executar Pipeline

<form action="/api/v1/pipelines/gitlab-qdrant/run" method="get" style="margin: 2rem 0;">
    <p style="margin-bottom: 1rem; color: #a8c5e0;">
        Inicie o processamento completo do pipeline ETL:
    </p>
    <button type="submit" name="action" value="run_pipeline">
        ▶️ Executar Pipeline ETL
    </button>
</form>

---

## 📊 Status Atual

<div id="pipeline-status" class="x-status-loading">
    <p>🔄 Carregando status...</p>
</div>

<script>
// Buscar status do pipeline
fetch('/api/v1/pipelines/gitlab-qdrant/run')
    .then(response => response.json())
    .then(data => {
        const statusDiv = document.getElementById('pipeline-status');
        const status = data.status;
        
        let emoji = '⏸️';
        let message = 'Pipeline parado';
        let color = '#8ca4c0';
        
        if (status === 'job_running') {
            emoji = '🚀';
            message = 'Pipeline em execução';
            color = '#5fbcd3';
        } else if (status === 'idle') {
            emoji = '✅';
            message = 'Pipeline pronto para execução';
            color = '#7db46c';
        }
        
        statusDiv.className = 'x-status';
        statusDiv.innerHTML = `
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 2rem;">${emoji}</span>
                <div>
                    <h3 style="color: ${color}; margin: 0;">${message}</h3>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #a8c5e0;">
                        Status: <code>${status}</code>
                    </p>
                </div>
            </div>
        `;
    })
    .catch(error => {
        document.getElementById('pipeline-status').innerHTML = `
            <p style="color: #e74c3c;">❌ Erro ao carregar status: ${error.message}</p>
        `;
    });
</script>

---

## 🏗️ Arquitetura do Sistema

O NIC ETL é construído com uma **arquitetura modular** baseada em notebooks Jupyter:

### 📋 Pipeline Completo

```mermaid
graph LR
    A[📥 GitLab] --> B[⚙️ Docling]
    B --> C[🔪 Chunks]
    C --> D[🧠 Embeddings]
    D --> E[💾 QDrant]
```

1. **🏗️ Fundação**: Configuração e validação do ambiente
2. **📥 GitLab**: Coleta de documentos do repositório
3. **⚙️ Docling**: Processamento e extração de conteúdo
4. **🔪 Chunks**: Segmentação inteligente de texto
5. **🧠 Embeddings**: Geração de vetores com IA
6. **💾 QDrant**: Armazenamento vetorial para busca semântica

## ✨ Características

<div class="x-features-grid">

<div class="x-feature-card">
    <span class="x-feature-icon">🔒</span>
    <div class="x-feature-title">Trava Inteligente</div>
    <p class="x-feature-description">
        Impede execuções simultâneas com sistema de lock automático
    </p>
</div>

<div class="x-feature-card">
    <span class="x-feature-icon">🌐</span>
    <div class="x-feature-title">Background Jobs</div>
    <p class="x-feature-description">
        Execução independente da conexão HTTP usando nohup
    </p>
</div>

<div class="x-feature-card">
    <span class="x-feature-icon">🧠</span>
    <div class="x-feature-title">IA Embeddings</div>
    <p class="x-feature-description">
        Vetorização avançada com modelo BAAI/bge-m3
    </p>
</div>

<div class="x-feature-card">
    <span class="x-feature-icon">📊</span>
    <div class="x-feature-title">Monitoramento</div>
    <p class="x-feature-description">
        Status em tempo real via JSON e logs detalhados
    </p>
</div>

</div>

## 🔗 Links Rápidos

- **[📊 Status Detalhado](/status)** - Monitoramento completo do pipeline
- **[🚀 Controle de Pipeline](/pipeline)** - Interface de execução
- **[📚 Documentação](/docs)** - Guias e referências
- **[🔗 API REST](/api/v1)** - Documentação OpenAPI

---

<div class="x-mission">
    <h3>🎯 Missão do NIC Lab</h3>
    <p>
        Transformar documentos em conhecimento acessível através de tecnologias de IA,
        facilitando a descoberta e o compartilhamento de informações organizacionais.
    </p>
</div>