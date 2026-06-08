🧙‍♂️ AnyoneDS — No-Code AutoML & Interactive SQL Studio

O AnyoneDS é uma plataforma inovadora de AutoML (Automated Machine Learning) e computação científica projetada para democratizar o acesso à ciência de dados. Com um design corporativo de alto contraste inspirado no Power BI Desktop e no Excel, o sistema permite a qualquer usuário subir planilhas avulsas, processar diagnósticos estatísticos imediatos, executar queries relacionais reais e treinar modelos preditivos de inteligência artificial com poucos cliques, sem digitar uma única linha de código.

🚀 Funcionalidades Principais

1. 🔬 Raio-X Estatístico & Data Wrangling Automatizado

Assim que uma planilha é carregada (suportando delimitadores americanos , ou o padrão brasileiro ; do Excel), o motor em Python analisa e limpa os dados brutos:

Diagnóstico Geral: Detecção em tempo real de total de linhas, colunas, contagem global de valores nulos (missing values), colunas numéricas e categóricas.

Estatística Descritiva Básica: Geração instantânea de mínimos, máximos, e médias aritméticas ignorando dados nulos e tratando exceções com segurança para colunas textuais.

2. 🧙‍♂️ Estúdio SQL & SQL Wizard de 10 Passos Dinâmico

Uma IDE relacional completa que funciona como um tutor interativo de banco de dados para iniciantes:

Compilador SQL Real: O arquivo CSV é carregado dinamicamente em uma base relacional SQLite em memória. Toda query digitada ou gerada é executada no banco e retorna em uma tabela de alto desempenho.

Jornada de 10 Passos: Um assistente inteligente em formato de balão que ensina a ordem exata de processamento lógico de dados ($SELECT$, $WHERE$, $GROUP\text{ }BY$, $HAVING$, $ORDER\text{ }BY$, $LIMIT$).

Toolbox de Funções Integradas: Atalhos para funções de Extração (CAST, SUBSTRING), Tratamento (TRIM, COALESCE, REPLACE, UPPER/LOWER), Análise (SUM, AVG, ROW_NUMBER() OVER()) e Validação (CASE WHEN, NULLIF).

3. 🤖 Laboratório de Machine Learning (AutoML)

Um pipeline preditivo acionado sob demanda de forma rápida e intuitiva:

Mapeamento de Problemas: Seleção rápida entre problemas de Regressão (previsão numérica contínua) ou Classificação (previsão categórica discreta).

Seleção Multimodelo: Escolha livre de algoritmos clássicos do Scikit-Learn: Regressão Linear, Regressão Logística, Árvores de Decisão e K-Nearest Neighbors (KNN).

Métricas Científicas e Validação:

Regressão: Cálculo instantâneo do Coeficiente de Determinação ($R^2$), Erro Médio Absoluto ($MAE$) e Raiz do Erro Quadrático Médio ($RMSE$).

Classificação: Acurácia, Precisão, Sensibilidade ($Recall$) e $F_1\text{-score}$.

Análise Gráfica Preditiva: Geração de gráficos dinâmicos comparando os valores Reais vs. Previstos obtidos na partição estocástica de testes (ajustável de $10\%$ a $50\%$).

📊 4. Visualização de Dados de Alto Contraste (Estilo Power BI)

Um painel de exploração visual projetado para máxima precisão e clareza de leitura:

Design de Alto Contraste: Interface limpa em fundo claro (Light Theme) para os gráficos com eixos, rótulos e linhas de grade nítidos, evitando qualquer sobreposição de texto.

Diversidade de Gráficos: Seleção interativa entre gráficos de Dispersão (Scatter), Linhas, Barras e Pizza.

Interatividade Científica: Tooltips flutuantes de alta fidelidade e legendas automáticas indicando as colunas e escalas que estão sendo plotadas.

🛠️ Stack Tecnológica

Frontend: React, TailwindCSS, Recharts (Visualizações Interativas) e Lucide/FontAwesome (Ícones).

Backend: Python (FastAPI), Pandas (Manipulação de Dados), NumPy (Álgebra Linear), Scikit-Learn (Pipelines de ML) e SQLite3 (Banco de Dados em Memória Relacional).

Hospedagem & Deploy: Render (Hospedagem do Microsserviço de Backend) e GitHub (Versionamento e Integração Contínua CI/CD).

🌍 Arquitetura do Sistema

O AnyoneDS adota um modelo de arquitetura desacoplada (Decoupled Architecture) de alta performance:

                  +---------------------------+
                  |     Interface React       |
                  |        (Lovable)          |
                  +-------------+-------------+
                                |
                 Envia CSV e    |  Retorna JSON de
                 Queries (POST) |  Métricas, Estatísticas
                                |  e Dados
                                v
                  +-------------+-------------+
                  |      API FastAPI          |
                  |       (Render)            |
                  +-------------+-------------+
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
     +---------------+  +---------------+  +---------------+
     |  Pandas/NumPy |  | Scikit-Learn  |  | SQLite Memory |
     | (Data Prep)   |  | (Modelos ML)  |  |  (Engine SQL) |
     +---------------+  +---------------+  +---------------+


Este desacoplamento garante que o navegador do usuário final permaneça leve e rápido, enquanto todo o processamento de álgebra linear e compilação relacional pesada acontecem de forma segura nos microsserviços do Render.
