# Generate Prompt

Estrutura uma ideia no formato ROCIN para posterior geração de PRP.

## Uso

```bash
/generate_prompt {descrição-da-ideia}
```

## Argumentos

`$ARGUMENTS` - Sua ideia completa, incluindo indicações de arquivos de exemplo

---

## Execução

### 1. Analisar entrada

Extrair da entrada:
- A ideia principal
- Menções a arquivos de exemplo (padrões: "exemplo em", "exemplos em", "example at", "examples in")
- Tecnologias mencionadas

### 2. Gerar nome do arquivo

Criar nome kebab-case curto e memorável baseado na ideia principal.
Exemplo: "qdrant-explorer", "jwt-auth", "tic-tac-toe"

### 3. Processar exemplos (se houver)

Se arquivos de exemplo forem mencionados:
- Copiar para `PRPs/Examples/{nome-do-prompt}/`
- Manter estrutura e nomes originais

### 4. Criar estrutura ROCIN

Gerar arquivo em `PRPs/Proposals/{nome}.md` com:

```markdown
# {Título baseado na ideia}

## Role
Act as a [papel apropriado baseado na ideia]

## Objective
[Reformular a ideia como objetivo claro e direto]

## Context
[Organizar informações da ideia original que definem escopo:
- Tecnologias mencionadas
- Funcionalidades descritas  
- Requisitos implícitos
- Referência aos exemplos copiados, se houver]

## Instructions
[Transformar a ideia em instruções claras:
- O que deve ser implementado
- Como deve funcionar
- Padrões a seguir baseados nos exemplos]

## Notes
[Qualquer detalhe adicional da ideia original que não se encaixou acima]
```

### Princípios importantes:

- **NÃO RESUMIR**: Manter todo conteúdo da ideia original
- **NÃO INVENTAR**: Apenas reorganizar o que foi fornecido
- **PRESERVAR DETALHES**: Cada aspecto mencionado deve aparecer no ROCIN
- **ESTRUTURAR**: Transformar texto corrido em estrutura clara

O objetivo é pegar exatamente o que o usuário disse e organizá-lo no formato ROCIN, sem adicionar pesquisa ou elaboração.