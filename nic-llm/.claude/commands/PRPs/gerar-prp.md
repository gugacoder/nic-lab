# gerar-prp.md

Gera um PRP baseado em um prompt por meio de pesquisa e análise detalhada

## Fase 1: Compreensão

* Ler o prompt de entrada $ARGUMENTS
* Se nenhum $ARGUMENTS for fornecido, ler `PRPs/PROMPT.md`
* Identificar o problema.
* Pedir esclarecimento ao usuário se necessário.
* Extrair e apresentar: objetivo em uma linha, escopo, restrições, premissas, critérios de sucesso, entregáveis.
* Apresentar em uma seção intitulada **“Declaração do Problema”**.

## Fase 2: Revisão de Artefatos do Projeto

* Escanear a pasta atual do projeto.
* Listar apenas arquivos, módulos, configs, convenções e padrões relevantes.
* Para cada item, apresentar caminho, tipo, relevância e observações.
* Apresentar em uma seção intitulada **“Artefatos e Convenções do Projeto”** como uma tabela.

## Fase 3: Revisão de Exemplos

* Escanear `PRPs/examples/` e subpastas.
* Identificar exemplos relevantes.
* Para cada um, apresentar caminho relativo, aspecto correspondente e notas de reutilização.
* Apresentar em uma seção intitulada **“Exemplos Aplicáveis”** como lista.

## Fase 4: Pesquisa

* Buscar fontes confiáveis na internet.
* Selecionar apenas fontes técnicas e autoritativas.
* Para cada uma, apresentar título, URL, publicador, principal aprendizado e aplicabilidade.
* Apresentar em uma seção intitulada **“Fontes Externas”** como tabela.

## Fase 5: Plano de Resolução

* Usar dados das fases 2–4.
* Pensar de forma modular para aumentar a coerência e reduzir a complexidade geral.
* Criar um plano sequencial.
* Cada passo deve incluir propósito, entradas, saída e verificação.
* Apresentar em uma seção intitulada **“Plano Passo a Passo”** como lista numerada.

## Fase 6: Geração do PRP

* Usar `PRPs/templates/prp.template.md`.
* Preencher completamente todas as seções aplicáveis do template. Descartar seções não aplicáveis.
* Incluir referências em três níveis: (1) artefatos do projeto, (2) exemplos, (3) fontes externas.
* Documentar *gotchas*: peculiaridades de bibliotecas, problemas de versão etc.
* Documentar padrões, boas práticas e snippets de código quando fontes mencionarem.
* Fornecer etapas explícitas de teste e aceitação com comandos ou casos reproduzíveis.
* Salvar o PRP em `PRPs/` como um novo arquivo.
* Apresentar o conteúdo completo do PRP inline.
* Adicionar uma seção intitulada **“Checklist de Execução”** no final do PRP.
* Nesta seção, apresentar tarefas como checklist markdown no formato `- [ ] ...`.
* As tarefas devem ser derivadas do problema específico, artefatos, exemplos, plano e requisitos do PRP.
* Nunca deixar o checklist vazio.

## Fase 7: Pontuação de Confiança

* Ao final do PRP, adicionar uma linha:
  `ConfidenceScore: X/5 — justificativa: [uma frase]`
* X deve seguir esta escala:

  * 1–2 = insuficiente
  * 3 = parcial e arriscado
  * 4 = suficiente
  * 5 = inequívoco

**Ao concluir, apresentar a pontuação de confiança.**

---

***CRITICAL: Pense somente após concluir a pesquisa e a exploração da base de código, e antes de iniciar a escrita do PRP.***

***ULTRATHINK: Reflita profundamente sobre o PRP, planeje a abordagem e só então comece a escrever.***
