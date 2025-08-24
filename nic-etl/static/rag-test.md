# 🔍 Teste de RAG - Busca Semântica

Interface interativa para testar o sistema de **RAG (Retrieval-Augmented Generation)** com busca híbrida.

## 🎯 Formulário de Consulta

<div class="x-feature-card" style="max-width: none; margin: 2rem 0;">
    <form id="rag-form" style="display: flex; flex-direction: column; gap: 1.5rem;">
        
        <!-- Consulta Principal -->
        <div>
            <label for="query" style="display: block; margin-bottom: 0.5rem; font-weight: 500; color: var(--color-blue);">
                📝 Consulta de Texto
            </label>
            <textarea 
                id="query" 
                name="query" 
                placeholder="Digite sua pergunta ou termo de busca..."
                rows="3"
                required
                style="
                    width: 100%; 
                    padding: 1rem; 
                    border: 2px solid var(--color-border); 
                    border-radius: 8px; 
                    background: var(--color-bg-secondary);
                    color: var(--color-text);
                    font-family: inherit;
                    font-size: 1rem;
                    resize: vertical;
                    outline: none;
                    transition: border-color 0.2s;
                "
            ></textarea>
        </div>

        <!-- Parâmetros de Busca -->
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
            <div>
                <label for="top_k" style="display: block; margin-bottom: 0.5rem; font-weight: 500; color: var(--color-blue);">
                    📊 Top K (Resultados)
                </label>
                <input 
                    type="number" 
                    id="top_k" 
                    name="top_k" 
                    value="20" 
                    min="1" 
                    max="50"
                    style="
                        width: 100%; 
                        padding: 0.75rem; 
                        border: 2px solid var(--color-border); 
                        border-radius: 6px; 
                        background: var(--color-bg-secondary);
                        color: var(--color-text);
                        font-family: inherit;
                    "
                />
            </div>
            
            <div>
                <label for="score_threshold" style="display: block; margin-bottom: 0.5rem; font-weight: 500; color: var(--color-blue);">
                    🎯 Score Mínimo
                </label>
                <input 
                    type="number" 
                    id="score_threshold" 
                    name="score_threshold" 
                    value="0.7" 
                    min="0" 
                    max="1" 
                    step="0.1"
                    style="
                        width: 100%; 
                        padding: 0.75rem; 
                        border: 2px solid var(--color-border); 
                        border-radius: 6px; 
                        background: var(--color-bg-secondary);
                        color: var(--color-text);
                        font-family: inherit;
                    "
                />
            </div>

            <div>
                <label for="include_embeddings" style="display: block; margin-bottom: 0.5rem; font-weight: 500; color: var(--color-blue);">
                    🧠 Incluir Embeddings
                </label>
                <select 
                    id="include_embeddings" 
                    name="include_embeddings"
                    style="
                        width: 100%; 
                        padding: 0.75rem; 
                        border: 2px solid var(--color-border); 
                        border-radius: 6px; 
                        background: var(--color-bg-secondary);
                        color: var(--color-text);
                        font-family: inherit;
                    "
                >
                    <option value="false">Não</option>
                    <option value="true">Sim</option>
                </select>
            </div>
        </div>

        <!-- Filtros de Metadata -->
        <div>
            <h4 style="margin: 0 0 1rem 0; color: var(--color-blue);">🏷️ Filtros de Metadata (Opcionais)</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <label for="repo" style="display: block; margin-bottom: 0.5rem; font-weight: 500;">
                        📁 Repositório
                    </label>
                    <input 
                        type="text" 
                        id="repo" 
                        name="repo" 
                        placeholder="nic/documentacao/base-de-conhecimento"
                        style="
                            width: 100%; 
                            padding: 0.75rem; 
                            border: 2px solid var(--color-border); 
                            border-radius: 6px; 
                            background: var(--color-bg-secondary);
                            color: var(--color-text);
                            font-family: inherit;
                        "
                    />
                </div>
                
                <div>
                    <label for="branch" style="display: block; margin-bottom: 0.5rem; font-weight: 500;">
                        🌿 Branch
                    </label>
                    <input 
                        type="text" 
                        id="branch" 
                        name="branch" 
                        placeholder="main"
                        style="
                            width: 100%; 
                            padding: 0.75rem; 
                            border: 2px solid var(--color-border); 
                            border-radius: 6px; 
                            background: var(--color-bg-secondary);
                            color: var(--color-text);
                            font-family: inherit;
                        "
                    />
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div>
                    <label for="source_document" style="display: block; margin-bottom: 0.5rem; font-weight: 500;">
                        📄 Documento
                    </label>
                    <input 
                        type="text" 
                        id="source_document" 
                        name="source_document" 
                        placeholder="Manual.pdf"
                        style="
                            width: 100%; 
                            padding: 0.75rem; 
                            border: 2px solid var(--color-border); 
                            border-radius: 6px; 
                            background: var(--color-bg-secondary);
                            color: var(--color-text);
                            font-family: inherit;
                        "
                    />
                </div>
                
                <div>
                    <label for="lang" style="display: block; margin-bottom: 0.5rem; font-weight: 500;">
                        🌍 Idioma
                    </label>
                    <select 
                        id="lang" 
                        name="lang"
                        style="
                            width: 100%; 
                            padding: 0.75rem; 
                            border: 2px solid var(--color-border); 
                            border-radius: 6px; 
                            background: var(--color-bg-secondary);
                            color: var(--color-text);
                            font-family: inherit;
                        "
                    >
                        <option value="">Todos</option>
                        <option value="pt-BR">Português</option>
                        <option value="en">English</option>
                    </select>
                </div>
            </div>
        </div>

        <!-- Botões -->
        <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 1rem;">
            <button 
                type="submit" 
                id="search-btn"
                style="
                    background: var(--color-blue); 
                    color: white; 
                    border: none; 
                    padding: 1rem 2rem; 
                    border-radius: 8px; 
                    font-size: 1rem; 
                    font-weight: 500; 
                    cursor: pointer; 
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                "
            >
                🔍 Buscar no RAG
            </button>
            
            <button 
                type="button" 
                id="clear-btn"
                style="
                    background: var(--color-gray); 
                    color: var(--color-text); 
                    border: 2px solid var(--color-border); 
                    padding: 1rem 2rem; 
                    border-radius: 8px; 
                    font-size: 1rem; 
                    cursor: pointer; 
                    transition: all 0.2s;
                "
            >
                🗑️ Limpar
            </button>
        </div>
    </form>
</div>

## 📊 Resultados da Busca

<div id="results-section" style="display: none; margin-top: 2rem;">
    
    <!-- Metadata da Busca -->
    <div id="search-metadata" class="x-info" style="margin-bottom: 2rem;">
        <h4>📈 Metadata da Consulta</h4>
        <div id="metadata-content"></div>
    </div>

    <!-- Lista de Resultados -->
    <div id="search-results">
        <h4>🔍 Resultados Encontrados</h4>
        <div id="results-list"></div>
    </div>

    <!-- Embeddings (se solicitado) -->
    <div id="embeddings-section" style="display: none; margin-top: 2rem;">
        <div class="x-config">
            <h4>🧠 Embeddings da Consulta</h4>
            <p>Vetor de embeddings gerado para a consulta (primeiras 10 dimensões):</p>
            <div id="embeddings-content" style="
                background: var(--color-bg-secondary); 
                padding: 1rem; 
                border-radius: 8px; 
                font-family: monospace; 
                font-size: 0.9rem;
                overflow-x: auto;
                border: 1px solid var(--color-border);
            "></div>
        </div>
    </div>
</div>

## 📚 Exemplos de Consulta

<div class="x-features-grid" style="margin-top: 2rem;">

<div class="x-example">
    <h5>🔍 Busca Simples</h5>
    <p><strong>Texto:</strong> "Quem é a Processa Sistemas?"</p>
    <p><strong>Top K:</strong> 20</p>
    <p><strong>Filtros:</strong> Nenhum</p>
</div>

<div class="x-example">
    <h5>🏷️ Busca com Filtros</h5>
    <p><strong>Texto:</strong> "processo de pagamento"</p>
    <p><strong>Branch:</strong> main</p>
    <p><strong>Idioma:</strong> pt-BR</p>
</div>

<div class="x-example">
    <h5>📄 Busca em Documento</h5>
    <p><strong>Texto:</strong> "identificação do cliente"</p>
    <p><strong>Documento:</strong> Manual.pdf</p>
    <p><strong>Embeddings:</strong> Sim</p>
</div>

</div>

<script>
// Funcionalidades específicas da página de teste RAG
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('rag-form');
    const searchBtn = document.getElementById('search-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultsSection = document.getElementById('results-section');
    const metadataContent = document.getElementById('metadata-content');
    const resultsList = document.getElementById('results-list');
    const embeddingsSection = document.getElementById('embeddings-section');
    const embeddingsContent = document.getElementById('embeddings-content');

    // Estilos dinâmicos para inputs
    document.querySelectorAll('input, textarea, select').forEach(input => {
        input.addEventListener('focus', function() {
            this.style.borderColor = 'var(--color-blue)';
            this.style.boxShadow = '0 0 0 2px rgba(95, 188, 211, 0.2)';
        });
        
        input.addEventListener('blur', function() {
            this.style.borderColor = 'var(--color-border)';
            this.style.boxShadow = 'none';
        });
    });

    // Botão hover effects
    searchBtn.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-2px)';
        this.style.boxShadow = '0 4px 12px rgba(95, 188, 211, 0.3)';
    });
    
    searchBtn.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
        this.style.boxShadow = 'none';
    });

    // Limpar formulário
    clearBtn.addEventListener('click', function() {
        form.reset();
        resultsSection.style.display = 'none';
        document.getElementById('top_k').value = '20';
        document.getElementById('score_threshold').value = '0.7';
        
        if (window.NICLab) {
            window.NICLab.Notifications.show('🗑️ Formulário limpo', 'info', 2000);
        }
    });

    // Submit do formulário
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Mostrar loading
        searchBtn.innerHTML = '⏳ Buscando...';
        searchBtn.disabled = true;
        resultsSection.style.display = 'none';
        
        try {
            // Coletar dados do formulário
            const formData = new FormData(form);
            const query = formData.get('query').trim();
            
            if (!query) {
                throw new Error('Por favor, digite uma consulta');
            }

            // Construir payload para a API
            const searchPayload = {
                query: query,
                top_k: parseInt(formData.get('top_k')) || 20,
                score_threshold: parseFloat(formData.get('score_threshold')) || 0.7,
                include_embeddings: formData.get('include_embeddings') === 'true'
            };

            // Adicionar filtros não vazios
            const filters = {};
            ['repo', 'branch', 'source_document', 'lang'].forEach(field => {
                const value = formData.get(field);
                if (value && value.trim()) {
                    filters[field] = value.trim();
                }
            });
            
            if (Object.keys(filters).length > 0) {
                searchPayload.filters = filters;
            }

            console.log('Enviando consulta:', searchPayload);

            // Fazer requisição para a API RAG com timeout estendido
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutos
            
            const response = await fetch('/rag/v1/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(searchPayload),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`Erro HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('Resultado da busca:', result);

            // Exibir resultados
            displayResults(result, searchPayload.include_embeddings);
            
            if (window.NICLab) {
                window.NICLab.Notifications.show(
                    `✅ Encontrados ${result.total_results} resultados`, 
                    'success'
                );
            }

        } catch (error) {
            console.error('Erro na busca:', error);
            
            let errorMessage = error.message;
            if (error.name === 'AbortError') {
                errorMessage = 'Consulta cancelada - tempo limite de 5 minutos excedido. Para consultas longas, isso é normal em laboratórios de pesquisa.';
            }
            
            if (window.NICLab) {
                window.NICLab.Notifications.show(
                    `❌ Erro: ${errorMessage}`, 
                    'error',
                    error.name === 'AbortError' ? 8000 : 5000
                );
            }
        } finally {
            // Restaurar botão
            searchBtn.innerHTML = '🔍 Buscar no RAG';
            searchBtn.disabled = false;
        }
    });

    function displayResults(result, includeEmbeddings) {
        // Metadata da busca
        metadataContent.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div>
                    <strong>📊 Total de Resultados:</strong><br>
                    <span style="color: var(--color-blue);">${result.total_results}</span>
                </div>
                <div>
                    <strong>⏱️ Tempo de Busca:</strong><br>
                    <span style="color: var(--color-blue);">${result.search_metadata.search_time_ms}ms</span>
                </div>
                <div>
                    <strong>🧠 Modelo:</strong><br>
                    <span style="color: var(--color-blue);">${result.search_metadata.model}</span>
                </div>
                <div>
                    <strong>💾 Collection:</strong><br>
                    <span style="color: var(--color-blue);">${result.search_metadata.collection}</span>
                </div>
                <div>
                    <strong>🎯 Score Threshold:</strong><br>
                    <span style="color: var(--color-blue);">${result.search_metadata.score_threshold}</span>
                </div>
                <div>
                    <strong>📈 Top K:</strong><br>
                    <span style="color: var(--color-blue);">${result.search_metadata.top_k}</span>
                </div>
            </div>
            
            ${Object.keys(result.search_metadata.filters_applied || {}).length > 0 ? `
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--color-border);">
                    <strong>🏷️ Filtros Aplicados:</strong><br>
                    <code style="background: var(--color-bg-secondary); padding: 0.5rem; border-radius: 4px; display: block; margin-top: 0.5rem;">
                        ${JSON.stringify(result.search_metadata.filters_applied, null, 2)}
                    </code>
                </div>
            ` : ''}
        `;

        // Lista de resultados
        if (result.results && result.results.length > 0) {
            resultsList.innerHTML = result.results.map((item, index) => `
                <div class="x-feature-card" style="margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: between; align-items: start; margin-bottom: 1rem;">
                        <h5 style="margin: 0; color: var(--color-blue);">
                            📄 Resultado ${index + 1}
                        </h5>
                        <div style="text-align: right;">
                            <span style="
                                background: var(--color-blue); 
                                color: white; 
                                padding: 0.25rem 0.75rem; 
                                border-radius: 20px; 
                                font-size: 0.9rem; 
                                font-weight: 500;
                            ">
                                Score: ${item.score}
                            </span>
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 1rem;">
                        <strong>📝 Texto:</strong>
                        <div style="
                            background: var(--color-bg-secondary); 
                            padding: 1rem; 
                            border-radius: 8px; 
                            margin-top: 0.5rem; 
                            border-left: 4px solid var(--color-blue);
                            font-size: 0.95rem;
                            line-height: 1.5;
                        ">
                            ${item.text.substring(0, 500)}${item.text.length > 500 ? '...' : ''}
                        </div>
                    </div>

                    ${item.highlights && item.highlights.length > 0 ? `
                        <div style="margin-bottom: 1rem;">
                            <strong>🎯 Destaques:</strong>
                            ${item.highlights.map(highlight => `
                                <div style="
                                    background: rgba(95, 188, 211, 0.1); 
                                    padding: 0.5rem; 
                                    border-radius: 4px; 
                                    margin-top: 0.25rem;
                                    font-size: 0.9rem;
                                    border-left: 3px solid var(--color-blue);
                                ">
                                    ${highlight}
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; font-size: 0.9rem;">
                        <div>
                            <strong>📁 Documento:</strong><br>
                            <span style="color: var(--color-blue);">${item.metadata.source_document || 'N/A'}</span>
                        </div>
                        <div>
                            <strong>📊 Chunk:</strong><br>
                            <span style="color: var(--color-blue);">${item.metadata.chunk_index || 'N/A'}</span>
                        </div>
                        <div>
                            <strong>🌿 Branch:</strong><br>
                            <span style="color: var(--color-blue);">${item.metadata.branch || 'N/A'}</span>
                        </div>
                        <div>
                            <strong>🆔 Point ID:</strong><br>
                            <span style="color: var(--color-blue);">${item.point_id || 'N/A'}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        } else {
            resultsList.innerHTML = `
                <div class="x-warning">
                    <h5>🔍 Nenhum resultado encontrado</h5>
                    <p>Tente ajustar sua consulta ou reduzir o score threshold.</p>
                </div>
            `;
        }

        // Embeddings (se solicitado)
        if (includeEmbeddings && result.query_embedding) {
            const embedding = result.query_embedding;
            const preview = embedding.slice(0, 10).map(val => val.toFixed(4)).join(', ');
            
            embeddingsContent.innerHTML = `
                <div style="margin-bottom: 1rem;">
                    <strong>📏 Dimensões:</strong> ${embedding.length}<br>
                    <strong>🔢 Preview (10 primeiras):</strong><br>
                    <span style="color: var(--color-blue);">[${preview}, ...]</span>
                </div>
                <details style="margin-top: 1rem;">
                    <summary style="cursor: pointer; font-weight: 500; color: var(--color-blue);">
                        🔍 Ver embedding completo
                    </summary>
                    <div style="
                        margin-top: 1rem; 
                        max-height: 300px; 
                        overflow-y: auto; 
                        padding: 1rem; 
                        background: var(--color-bg); 
                        border-radius: 6px; 
                        word-break: break-all;
                        font-size: 0.8rem;
                    ">
                        ${JSON.stringify(embedding, null, 2)}
                    </div>
                </details>
            `;
            embeddingsSection.style.display = 'block';
        } else {
            embeddingsSection.style.display = 'none';
        }

        // Mostrar seção de resultados
        resultsSection.style.display = 'block';
        
        // Scroll suave para os resultados
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
</script>

---

<div class="x-tip">
    <h4>💡 Dicas de Uso</h4>
    <ul>
        <li><strong>Score Threshold:</strong> Valores mais baixos (0.5-0.7) retornam mais resultados, valores mais altos (0.8-0.9) são mais restritivos</li>
        <li><strong>Top K:</strong> Controla quantos resultados são retornados (máximo recomendado: 50)</li>
        <li><strong>Filtros:</strong> Use filtros de metadata para refinar a busca por repositório, branch ou documento específico</li>
        <li><strong>Embeddings:</strong> Ative para ver o vetor numérico usado na busca semântica</li>
    </ul>
</div>