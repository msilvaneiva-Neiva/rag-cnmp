# Sistema Inteligente de Consulta ao Prêmio CNMP (RAG)

**Trabalho Prático Final — Opção 2: Desenvolvimento de um Sistema Inteligente Baseado em LLM para Consulta e Análise de Documentos**

**Disciplina:** Engenharia de Software para Modelos de IA
**Professor:** Altino Dantas
**Equipe:** Carlos André Martins Neves, Helena Rocha Fernandes, Marina da Silva Neiva
**Data de entrega:** 03/08/2026

## 1. Objetivo

Este projeto implementa uma aplicação de Inteligência Artificial capaz de responder, em linguagem natural, perguntas sobre uma base documental específica, utilizando um pipeline completo de RAG (Retrieval-Augmented Generation): coleta e preparação dos documentos, geração de embeddings, armazenamento em banco vetorial, recuperação de contexto, integração com um LLM e interface de consulta.

O domínio escolhido foi o **Prêmio CNMP**, iniciativa do Conselho Nacional do Ministério Público (CNMP) que reconhece programas e projetos de destaque do Ministério Público brasileiro. A base documental cobre as edições de 2013 a 2026.

## 2. Descrição do Contexto dos Documentos

A base é composta por **67 documentos** em formato PDF, TXT e HTML, coletados diretamente do Google Drive da equipe (pasta "CNMP"), organizados por ano de edição do prêmio (2013–2026) e por uma pasta de documentos-piloto da edição de 2025. Os tipos de documento incluem:

- **Normativos**: resoluções, portarias e regulamentos do CNMP que instituem e regulam o Prêmio CNMP ao longo dos anos.
- **Listas de resultado**: arquivos `Premiados.txt` (um por ano, 2013–2025) contendo a relação oficial de projetos vencedores por categoria e colocação (1º, 2º, 3º lugar).
- **Listas de processo seletivo**: PDFs com iniciativas pré-habilitadas, habilitadas, semifinalistas e finalistas (edição 2025), documentando as etapas do processo de seleção.
- **Materiais institucionais**: livretos de vencedores, cronogramas, catálogos de projetos e catálogos HTML interativos.

Dos 67 documentos coletados, **todos os 67 foram extraídos com sucesso (100%)**: 65 diretamente (`pypdf`/`BeautifulSoup`) e 2 PDFs escaneados sem camada de texto (um livreto de 2022 e uma resolução de 2013) recuperados via OCR (Tesseract, em português).

Este domínio foi escolhido por se enquadrar diretamente na categoria "documentação técnica / regulamentos / relatórios" sugerida no enunciado, e por oferecer um desafio real de recuperação de informação: os documentos combinam texto normativo (regulamentos), texto tabular/estruturado (listas de premiados) e texto narrativo (livretos), exigindo uma estratégia de busca robusta.

## 3. Arquitetura do Sistema

```
Documentos (Drive)
        │
        ▼
[Coleta] Google Drive API (chave pública) ──► data/CNMP_raw/ ──► filtro por extensão ──► data/CNMP_docs/
        │
        ▼
[Preparação] extração de texto (pypdf / BeautifulSoup) + OCR de fallback (Tesseract) ──► chunking (RecursiveCharacterTextSplitter)
        │
        ▼
[Indexação] embeddings (sentence-transformers) ──► ChromaDB (persistente)
        │
        ▼
[Consulta] pergunta do usuário (+ histórico da conversa)
        │
        ▼
[Recuperação] busca híbrida: vetorial (ChromaDB) + lexical (BM25) + boost em "Premiados.txt"
        │
        ▼
[Geração] LLM (Claude Sonnet 5, via OpenRouter) gera resposta com base no contexto recuperado e no histórico
        │
        ▼
[Interface] Gradio (chat) exibe resposta + fontes utilizadas
```

## 4. Processo de Preparação dos Dados

### 4.1 Coleta

Os documentos foram baixados programaticamente via **Google Drive API** (autenticação por chave de API pública, sem necessidade de login individual), percorrendo recursivamente a estrutura de pastas do Drive, e em seguida filtrados para manter apenas arquivos `.pdf`, `.txt`, `.html` e `.htm` (script `scripts/download_documents.py`). Essa abordagem substituiu uma tentativa inicial com `gdown --folder`, que se mostrou instável por bloqueios anti-automação do Google Drive quando muitos arquivos são baixados anonimamente em sequência.

### 4.2 Extração de texto

Cada tipo de arquivo é tratado por um extrator específico (`src/document_loader.py`):

- **PDF**: `pypdf.PdfReader`, concatenando o texto de todas as páginas; quando a extração retorna vazio (PDF escaneado), aciona automaticamente um fallback de **OCR** (`pytesseract` + `pdf2image`, em português).
- **TXT**: leitura direta em UTF-8.
- **HTML**: `BeautifulSoup`, removendo tags e extraindo apenas o texto visível.

Documentos que ainda assim retornam texto vazio são registrados como aviso e excluídos da base indexada, sem interromper o processamento dos demais.

### 4.3 Chunking (segmentação)

Utilizamos o `RecursiveCharacterTextSplitter` do LangChain, com `chunk_size=1000` e `chunk_overlap=150` caracteres, priorizando quebras em parágrafos e frases antes de cortar no meio de palavras. O processo gerou **1.258 chunks** a partir dos 67 documentos extraídos, cada um com metadado de origem (`source`) e um identificador único (`chunk_id`).

## 5. Embeddings Utilizados

Optamos pelo modelo **paraphrase-multilingual-MiniLM-L12-v2** (biblioteca `sentence-transformers`), rodando localmente no ambiente (sem custo de API). A escolha se deu por três motivos: (1) suporte nativo a português, essencial para um corpus 100% em português; (2) tamanho reduzido (leve o suficiente para rodar em CPU no Google Colab gratuito); (3) independência de uma chave de API adicional só para embeddings.

## 6. Banco Vetorial

Utilizamos **ChromaDB** em modo `PersistentClient`, com uma única coleção (`cnmp_docs`) contendo os 1.258 chunks indexados em lotes de 100. A escolha do ChromaDB se deu pela simplicidade de configuração (não exige infraestrutura externa) e pela integração nativa com `sentence-transformers` via `embedding_functions`.

## 7. Recuperação de Contexto: Busca Híbrida (Funcionalidade Bônus 1)

Durante os testes, identificamos uma limitação real da busca puramente vetorial: perguntas envolvendo termos exatos (siglas de unidades do Ministério Público, como "MP/GO", ou categorias específicas como "Tecnologia da Informação") frequentemente não recuperavam os chunks corretos, especialmente quando a informação estava em conteúdo tabular (as listas em `Premiados.txt`), que embeda de forma menos "semântica" que texto narrativo.

Para resolver esse problema — e simultaneamente atender ao item de bônus **"Busca híbrida (vetorial + lexical)"** — implementamos em `src/retrieval.py` uma estratégia de recuperação em três camadas:

1. **Busca vetorial** (ChromaDB / embeddings semânticos), captura similaridade de significado.
2. **Busca lexical (BM25)**, via `rank_bm25`, captura correspondência exata de termos.
3. **Reciprocal Rank Fusion (RRF)** combina os rankings das duas buscas em um único score.
4. **Boost heurístico**: quando a pergunta contém termos indicativos de resultado ("venceu", "ganhou", "premiação", "1º lugar" etc.), o sistema faz uma busca literal adicional por siglas e nomes próprios extraídos da pergunta (regex) diretamente nos arquivos `Premiados.txt`, garantindo que a fonte oficial de resultados seja sempre considerada quando relevante.

Esse ajuste foi motivado por um caso real de teste (documentado na íntegra no notebook de desenvolvimento): a pergunta "Quais premiações o MP/GO ganhou?" inicialmente retornava respostas incompletas ou incorretas; após a implementação da busca híbrida com boost, o sistema passou a responder corretamente, listando 17 premiações do MP/GO distribuídas em 6 edições (2014, 2015, 2019, 2022, 2023 e 2025), citando 21 fontes corretamente identificadas — incluindo um caso em que o próprio sistema identificou e sinalizou uma inconsistência entre os relatórios de tendências e os dados detalhados por ano.

## 8. Histórico de Conversas (Funcionalidade Bônus 2)

Implementamos memória de conversa para permitir perguntas de acompanhamento em linguagem natural (ex.: "e em 2014, quem venceu essa mesma categoria?"). A função `answer_question_with_history` (`src/rag.py`) incorpora as últimas trocas da conversa no prompt enviado ao LLM, tratando de forma defensiva tanto o formato de histórico em dicionários (`{"role": ..., "content": ...}`) quanto o formato legado em tuplas `(pergunta, resposta)`, garantindo compatibilidade independentemente da versão do Gradio instalada.

A interface de chat inclui ainda filtros complementares (MP e ano) que são incorporados à pergunta enviada ao pipeline, refinando o contexto recuperado sem exigir alterações na lógica de busca. Após cada resposta, os filtros são reiniciados automaticamente para a próxima pergunta.

## 9. LLM Utilizado

Utilizamos o **Claude Sonnet 5** (Anthropic), acessado via **OpenRouter** (`anthropic/claude-sonnet-5`), com `temperature=0.2` para respostas mais determinísticas e factuais. O prompt instrui explicitamente o modelo a responder apenas com base no contexto recuperado e a admitir quando a informação não está disponível, minimizando alucinações — comportamento confirmado em múltiplos testes (ex.: quando perguntado sobre uma categoria não presente no contexto recuperado, o sistema respondeu "Não encontrei essa informação nos documentos fornecidos" em vez de inventar uma resposta).

## 10. Interface de Consulta

A interface foi construída com **Gradio** (`gr.ChatInterface`), rodando diretamente no Google Colab com link público temporário (`share=True`). A interface permite:

- Inserção de perguntas em linguagem natural, com memória das trocas anteriores da conversa;
- Filtros complementares por MP e ano, reiniciados automaticamente após cada resposta;
- Exibição da resposta gerada pelo LLM;
- Exibição separada das fontes (documentos) utilizadas para gerar a resposta, atendendo ao requisito de transparência da recuperação.

## 11. Engenharia de Software

### 11.1 Organização em módulos

O código de produção está organizado em pacote Python (`src/`), com responsabilidades separadas:

| Módulo | Responsabilidade |
|---|---|
| `config.py` | Configurações centrais (variáveis de ambiente, constantes) |
| `document_loader.py` | Extração de texto (PDF/TXT/HTML), com fallback de OCR |
| `chunking.py` | Segmentação dos textos |
| `vectorstore.py` | Geração de embeddings e indexação no ChromaDB |
| `retrieval.py` | Busca híbrida (vetorial + BM25 + boost) |
| `rag.py` | Orquestração do pipeline RAG, histórico de conversa e chamada ao LLM |
| `app.py` (raiz) | Ponto de entrada: monta o pipeline e sobe a interface Gradio |
| `scripts/download_documents.py` | Reprodução da etapa de coleta dos documentos via Drive API |

### 11.2 Controle de versão

O projeto está versionado em Git, com histórico de commits, e publicado no GitHub: **https://github.com/msilvaneiva-Neiva/rag-cnmp**.

### 11.3 Configuração por variável de ambiente

As chaves de API (OpenRouter e Google Drive) são carregadas via arquivo `.env` (não versionado, protegido por `.gitignore`), usando `python-dotenv`. Um arquivo `.env.example` documenta as variáveis necessárias (`OPENROUTER_API_KEY` e `DRIVE_API_KEY`) para qualquer pessoa reproduzir o ambiente.

### 11.4 Tratamento de erros

- A extração de documentos trata falhas por arquivo individualmente (`try/except`), sem interromper o processamento em lote, registrando avisos para arquivos vazios ou ilegíveis.
- A coleta de documentos (Drive API) trata falhas por arquivo individualmente, registrando e reportando ao final quais falharam, sem interromper o download dos demais.
- A função de resposta do RAG (`RAGEngine.answer`) captura exceções (falhas de rede, de API, etc.) e retorna uma mensagem de erro amigável em vez de quebrar a aplicação.
- A verificação de variáveis de ambiente ausentes (`config.py`) falha de forma explícita e informativa, orientando o usuário a configurar o `.env`.

## 12. Processo de Desenvolvimento e Ajustes

Um diferencial deste projeto foi o processo de teste e refinamento iterativo, documentado célula a célula no notebook de desenvolvimento (`notebook_desenvolvimento.ipynb`, incluído no repositório):

1. A coleta inicial via `gdown --folder` funcionou parcialmente, mas passou a falhar por bloqueio anti-automação do Google Drive em execuções com muitos downloads sequenciais. Migramos para a Google Drive API com chave pública, com pausas entre requisições e retomada automática de arquivos já baixados.
2. A primeira versão do pipeline de busca (vetorial pura) funcionou bem para perguntas com vocabulário próximo ao dos documentos, mas falhou em perguntas específicas sobre resultados por categoria/ano.
3. Diagnosticamos, arquivo por arquivo, que os dados estavam corretos e indexados na base — o problema era de *ranking* na recuperação, não de dados ausentes.
4. Implementamos busca híbrida (vetorial + BM25), que resolveu parte dos casos, mas ainda falhava para siglas de unidades (ex.: "MP/GO"), pois tokens curtos como "mp" e "go" são pouco discriminativos em um corpus onde toda unidade é referenciada como "MP/XX".
5. Adicionamos uma camada de busca literal por entidades nomeadas (siglas e nomes próprios) especificamente nos arquivos de resultado, resolvendo definitivamente o problema.
6. Dois PDFs escaneados sem camada de texto foram identificados e recuperados com sucesso via OCR (Tesseract).
7. Implementamos a funcionalidade de histórico de conversas, com tratamento defensivo para os diferentes formatos de histórico usados por versões distintas do Gradio.

Essa jornada está documentada em detalhe no notebook, incluindo os testes que falharam, os diagnósticos executados e as correções aplicadas — evidenciando um processo real de engenharia e depuração, não apenas um pipeline que funcionou de primeira.

## 13. Como Executar o Projeto

```bash
git clone https://github.com/msilvaneiva-Neiva/rag-cnmp.git
cd rag-cnmp
pip install -r requirements.txt
cp .env.example .env
# edite o .env e insira sua OPENROUTER_API_KEY e DRIVE_API_KEY

python scripts/download_documents.py   # baixa e filtra a base documental
python app.py                          # monta o pipeline e sobe a interface Gradio
```

Alternativamente, o notebook de desenvolvimento (`notebook_desenvolvimento.ipynb`) pode ser executado célula a célula no Google Colab, reproduzindo todo o processo de forma incremental e documentada.

## 14. Limitações e Trabalhos Futuros

- **Ambiguidade de siglas**: apesar do boost implementado, siglas muito curtas ou ambíguas ainda podem exigir reformulação da pergunta pelo usuário para obter melhor precisão.
- **Dependência de disponibilidade da API do Google Drive**: embora mais robusta que a abordagem anônima inicial, a coleta ainda depende da disponibilidade da Drive API; uma cópia local/cache da base bruta mitigaria esse risco em reexecuções futuras.
- **Deploy**: a interface atual usa um link temporário do Gradio (válido por até 7 dias). Para uso contínuo, seria necessário um deploy permanente (ex.: Hugging Face Spaces ou um serviço de nuvem).

## 15. Exemplo de Uso

**Pergunta:** "Quais premiações o MP/GO (Ministério Público de Goiás) ganhou no Prêmio CNMP?"

**Resposta (resumo):** O sistema retornou corretamente 17 premiações do MP/GO em 6 edições (2014, 2015, 2019, 2022, 2023 e 2025), distribuídas em categorias como Defesa dos Direitos Fundamentais, Diminuição da Corrupção, Tecnologia da Informação e Gestão e Governança, citando 21 fontes — incluindo uma observação espontânea do sistema sobre uma inconsistência entre os relatórios de tendências e os dados detalhados por ano, demonstrando a efetividade da busca híbrida implementada.

## 16. Conclusão

O projeto atende a todos os requisitos mínimos do enunciado (coleta e preparação de mais de 20 documentos, geração de embeddings, banco vetorial, RAG, integração com LLM, interface de consulta com exibição de fontes, organização modular do código, controle de versão, configuração por ambiente e tratamento de erros) e implementa **duas** funcionalidades bônus — **busca híbrida (vetorial + lexical)** e **histórico de conversas** —, ambas motivadas por limitações reais identificadas durante o desenvolvimento, evidenciando um ciclo completo de engenharia de software aplicada a modelos de IA: implementar, testar, diagnosticar e corrigir.
