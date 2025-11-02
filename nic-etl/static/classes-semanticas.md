# 🎨 Classes Semânticas - Guia de Uso

Este documento detalha as classes semânticas disponíveis para uso em conteúdo markdown, substituindo estilos inline por CSS semântico.

## 📋 Classes Disponíveis

### 📊 Status e Indicadores

#### `.x-status`
Para exibir status de sistema ou pipeline.

```html
<div class="x-status">
    <p>Status do sistema atual</p>
</div>
```

#### `.x-status-loading`
Para exibir estados de carregamento.

```html
<div class="x-status-loading">
    <p>🔄 Carregando dados...</p>
</div>
```

### 🎯 Cards e Grids

#### `.x-features-grid`
Grid responsivo para exibir características/features.

```html
<div class="x-features-grid">
    <!-- Cards aqui -->
</div>
```

#### `.x-feature-card`
Cards individuais para features.

```html
<div class="x-feature-card">
    <span class="x-feature-icon">🚀</span>
    <div class="x-feature-title">Título da Feature</div>
    <p class="x-feature-description">Descrição da feature</p>
</div>
```

### 💬 Mensagens e Alertas

#### `.x-alerta`
Para alertas críticos (adiciona ⚠️ automaticamente).

<div class="x-alerta">
Este é um alerta importante que requer atenção imediata.
</div>

```html
<div class="x-alerta">
Este é um alerta importante que requer atenção imediata.
</div>
```

#### `.x-warning`
Para avisos de cuidado (adiciona ⚠️ automaticamente).

<div class="x-warning">
Este é um aviso de cuidado para o usuário.
</div>

```html
<div class="x-warning">
Este é um aviso de cuidado para o usuário.
</div>
```

#### `.x-success`
Para mensagens de sucesso (adiciona ✅ automaticamente).

<div class="x-success">
Operação realizada com sucesso!
</div>

```html
<div class="x-success">
Operação realizada com sucesso!
</div>
```

#### `.x-tip`
Para dicas úteis (adiciona 💡 automaticamente).

<div class="x-tip">
Esta é uma dica útil para melhorar sua experiência.
</div>

```html
<div class="x-tip">
Esta é uma dica útil para melhorar sua experiência.
</div>
```

#### `.x-note`
Para notas informativas (adiciona 📝 automaticamente).

<div class="x-note">
Esta é uma nota informativa importante.
</div>

```html
<div class="x-note">
Esta é uma nota informativa importante.
</div>
```

### 📚 Conteúdo Especializado

#### `.x-info`
Para caixas de informação gerais.

<div class="x-info">
    <h3>Informação Importante</h3>
    <p>Conteúdo da informação aqui.</p>
</div>

```html
<div class="x-info">
    <h3>Informação Importante</h3>
    <p>Conteúdo da informação aqui.</p>
</div>
```

#### `.x-mission`
Para destacar missão, visão ou mensagens principais.

<div class="x-mission">
    <h3>🎯 Nossa Missão</h3>
    <p>Descrição da missão da organização.</p>
</div>

```html
<div class="x-mission">
    <h3>🎯 Nossa Missão</h3>
    <p>Descrição da missão da organização.</p>
</div>
```

#### `.x-example`
Para exemplos de código ou uso (adiciona "📋 Exemplo:" automaticamente).

<div class="x-example">
# Exemplo de configuração
export API_KEY="sua-chave-aqui"
export DATABASE_URL="postgresql://..."
</div>

```html
<div class="x-example">
# Exemplo de configuração
export API_KEY="sua-chave-aqui"
export DATABASE_URL="postgresql://..."
</div>
```

#### `.x-config`
Para blocos de configuração (adiciona ⚙️ automaticamente).

<div class="x-config">
Configure os seguintes parâmetros no arquivo .env antes de executar o sistema.
</div>

```html
<div class="x-config">
Configure os seguintes parâmetros no arquivo .env antes de executar o sistema.
</div>
```

#### `.x-pipeline-stage`
Para etapas de pipeline (adiciona 🚀 automaticamente).

<div class="x-pipeline-stage">
Esta etapa processa os documentos usando OCR e extração de texto.
</div>

```html
<div class="x-pipeline-stage">
Esta etapa processa os documentos usando OCR e extração de texto.
</div>
```

## 🎨 Características das Classes

### ✨ Funcionalidades Automáticas

1. **Ícones Automáticos**: Muitas classes adicionam ícones automaticamente
2. **Temas Adaptativos**: Todas as classes se adaptam aos temas claro/escuro
3. **Responsive**: Design responsivo automático
4. **Hover Effects**: Efeitos de hover onde apropriado
5. **Transições Suaves**: Animações CSS para mudanças de tema

### 🎯 Diretrizes de Uso

1. **Semântica**: Use a classe que melhor representa o significado do conteúdo
2. **Consistência**: Mantenha consistência no uso das classes em todo o projeto
3. **Hierarquia**: Use `.x-info` para informações gerais, `.x-tip` para dicas específicas
4. **Contexto**: Escolha a classe baseada no contexto (alerta vs aviso vs nota)

### 🔧 Customização

Todas as classes usam CSS Variables e se adaptam automaticamente aos temas:

- **Tema Escuro**: Cores mais escuras e contrastes adequados
- **Tema Claro**: Cores mais claras e contrastes suaves
- **Transições**: Mudanças suaves entre temas

## 📖 Exemplos Práticos

### Pipeline ETL

<div class="x-pipeline-stage">
**Etapa 1**: Coleta de documentos do GitLab
</div>

<div class="x-pipeline-stage">
**Etapa 2**: Processamento com Docling
</div>

<div class="x-config">
Configure sua API key do GitLab no arquivo `.env`
</div>

### Documentação

<div class="x-tip">
Execute `npm install` antes de iniciar o desenvolvimento.
</div>

<div class="x-warning">
Não execute este comando em produção sem backup.
</div>

<div class="x-success">
Sistema configurado e pronto para uso!
</div>

---

💡 **Dica**: Prefira sempre classes semânticas a estilos inline para melhor manutenibilidade e consistência visual.