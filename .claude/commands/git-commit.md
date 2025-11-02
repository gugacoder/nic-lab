# git-commit

## Objetivo

Registrar mudanças com mensagens de commit concisas, padronizadas e inequívocas, seguindo o modelo `conventional commits`.

## Instruções

Leia $ARGUMENTS como opções.

**Modo de commit**: Se a opção `-all` for fornecida, habilite o modo full-commit; caso contrário, habilite o modo stage-only.

**Validação de stage**: Se o modo full-commit estiver habilitado, adicione todos os arquivos da pasta atual e suas subpastas ao stage, equivalente a `git add -A .`. Se o modo stage-only estiver habilitado, faça commit apenas do que já está no stage.

**Mensagem de commit**: Declare claramente **o que foi alterado e por quê**, de forma legível tanto para humanos quanto para máquinas, considerando apenas mudanças nos arquivos já adicionados ao stage.

**Ferramentas extras**: Use o TodoWrite para orientar a execução.

## Estrutura

```bash
<tipo>(<escopo>): <resumo direto da mudança>

- [opcional] bullets com decisões ou exemplos
```

## Tipos comuns

* `feat`: Adiciona uma nova funcionalidade
* `fix`: Corrige um bug
* `docs`: Alterações na documentação
* `style`: Mudanças de estilo de código (espaços, ponto e vírgula, etc.) sem impacto funcional
* `refactor`: Refatoração de código sem alterar funcionalidade
* `perf`: Melhorias de performance
* `test`: Adição ou atualização de testes
* `build`: Mudanças que afetam o sistema de build ou dependências
* `ci`: Alterações em arquivos e scripts de CI
* `chore`: Outras mudanças que não modificam src ou testes
* `revert`: Reverte um commit anterior

## Escopos recomendados

* `system`, `commands`, `blueprint`, `template`
* `domain`, `feature`, `task`
* `docs` (quando transversal)

## Exemplos

```bash
docs(system): converter CLAUDE.md para README.md

- Removidas referências a IA
- Padronizado com a estrutura DFT
```

```bash
feat(feature): adicionar comentários ao blog
fix(task): corrigir bug na exportação JSON
```

## Restrições

* Não descreva o passo a passo — declare **o resultado final**
* Evite termos vagos como "ajustes" ou "melhorias"
* Não inclua explicações fora da mensagem de commit

## Dica

Se a mudança não for testável, o commit deve ser ao menos **rastreável e atômico**.
