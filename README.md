# Avaliação Comparativa de Modelos de IA — Sistema Especialista, Árvore de Decisão e Ensembles

**Aluno:** [preencher com nome completo]  
**Matrícula:** [preencher com a matrícula]  
**Dataset:** [link direto para o dataset utilizado]  
**Repositório:** [link público do GitHub/GitLab]

## Descrição

Este repositório compara três abordagens de modelagem — **sistema especialista
baseado em regras**, **árvore de decisão** e **ensembles** (Random Forest e
Gradient Boosting) — em dois problemas do mesmo domínio de risco de diabetes:

- **Classificação:** prever `Diabetes_Risk` nas classes `Low`, `Moderate` e `High`.
- **Regressão:** prever o escore contínuo `Diabetes_Risk_Score` (0–100).

A avaliação segue as métricas da Aula 08: desempenho preditivo, generalização
(gap treino/teste e validação cruzada de 5 folds), eficiência computacional
(tempo, memória, tamanho), energia/CO₂ e análise comparativa final com
recomendação ao cliente.

O diferencial do trabalho é tratar o sistema especialista como um modelo
avaliável, e não apenas como uma descrição qualitativa: as regras são
executadas por um motor de inferência vetorizado, comparadas aos modelos de ML
nos mesmos conjuntos e cronometradas em sessões registradas em `logs/`.

## Estrutura do repositório

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                         # dataset original (não versionado; ver link)
│   └── processed/
│       ├── dataset_classification.csv
│       └── dataset_regression.csv
├── notebooks/
│   ├── 03_expert_system.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_ensemble.ipynb
│   ├── 06_evaluation_metrics.ipynb
│   ├── 07_generalization.ipynb
│   ├── 08_computational_efficiency.ipynb
│   ├── 09_comparative_analysis.ipynb
│   └── archive/                     # notebooks exploratórios das etapas anteriores
├── src/
│   ├── config.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── expert_system.py
│   ├── models.py
│   ├── metrics.py
│   ├── profiling.py
│   ├── interpretability.py
│   ├── rules_io.py
│   ├── validate_data.py
│   └── visualization.py
├── rules/
│   ├── regras_classificacao.py/.txt
│   ├── regras_regressao.py/.txt
│   └── regras_regressao_v1.txt
├── logs/
│   ├── cronometro_regras.csv
│   ├── esforco_humano.csv
│   └── README.md
├── reports/
│   ├── tables/                      # CSVs de resultados
│   └── figures/                     # gráficos em PNG
├── artifacts/
│   └── models/                      # modelos treinados (.joblib)
└── docs/                            # relatório final (PDF e HTML)
```

## Como executar

Pré-requisitos: Python 3.11+ (o ambiente usado tinha Python 3.14.3) e as
dependências de `requirements.txt`.

```bash
# 1. Clone o repositório
git clone [URL_DO_REPOSITORIO]
cd avaliacao-comparativa-modelos-ia

# 2. Crie e ative um ambiente virtual
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Registre o kernel usado pelos notebooks
python -m ipykernel install --user --name venv-ia --display-name "Python (venv IA)"
```

Os datasets processados já estão em `data/processed/`. Se necessário, gere-os a
partir do dataset original seguindo as etapas de EDA/pré-processamento da
atividade anterior.

Execute os notebooks **nesta ordem**, sem pular etapas:

1. `03_expert_system.ipynb` — regras, motor de inferência e medições.
2. `04_decision_tree.ipynb` — árvore de decisão e ajuste de hiperparâmetros.
3. `05_ensemble.ipynb` — Random Forest, Gradient Boosting e importâncias.
4. `06_evaluation_metrics.ipynb` — métricas, ROC/PR, matrizes e resíduos.
5. `07_generalization.ipynb` — gap, validação cruzada e viés–variância.
6. `08_computational_efficiency.ipynb` — tempo, memória, energia e CO₂.
7. `09_comparative_analysis.ipynb` — tabela final, perguntas (a)–(f) e recomendação.

Exemplo de execução em lote (após ativar o ambiente):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=venv-ia notebooks/03_expert_system.ipynb
```

Repita para os demais notebooks. A ordem importa porque os notebooks 04–09
consomem artefatos e tabelas gerados pelos anteriores.

## Ambiente computacional

| Item | Valor |
|---|---|
| Processador | 13th Gen Intel Core i5-13420H, 8 núcleos / 12 threads |
| Memória RAM | 7,7 GB |
| Sistema operacional | Windows 11 (build 26200) |
| Python | 3.14.3 |
| pandas | 3.1.0 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| joblib | 1.6.0 |

Semente global: `RANDOM_STATE = 42`. Os tempos dependem do hardware e não devem
ser comparados diretamente com medições de outras máquinas.

## Principais resultados

A tabela completa, com desempenho, gap, validação cruzada e eficiência, está em
[`reports/tables/resumo_final.csv`](reports/tables/resumo_final.csv).

### Classificação — F1-Score macro

| Modelo | F1 teste | CV 5-fold | Gap F1 | Accuracy | Tamanho | Inferência |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | **0,6862** | 0,6838 ± 0,0247 | 0,3138 | 0,8678 | 69,1 MiB | 0,0097 ms/amostra |
| Gradient Boosting | 0,6681 | **0,7126 ± 0,0200** | 0,0826 | **0,8790** | 775 KiB | 0,0088 ms/amostra |
| Árvore de Decisão | 0,6649 | 0,6781 ± 0,0300 | 0,0823 | 0,8595 | 131 KiB | **0,0019 ms/amostra** |
| Especialista (cobertura 53,87%) | 0,5618 | 0,5687 ± 0,0122 | **0,0070** | 0,8766 | **0,8 KiB** | 0,0049 ms/amostra |

A Random Forest lidera o F1 em uma única divisão, mas a diferença para o
Gradient Boosting (0,0181) é menor que o desvio padrão da CV. O GB tem a melhor
média de CV, a melhor accuracy e um artefato ~91× menor que a RF. O especialista
tem o melhor F1 da classe `High` (0,9539), mas deixa 46,13% dos registros sem
classificação.

### Regressão — MAE

| Modelo | MAE teste | CV 5-fold | Gap MAE | R² teste | Tamanho | Inferência |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | **5,9278** | **5,9188 ± 0,0316** | 0,0461 | **0,7745** | 157 KiB | 0,0031 ms/amostra |
| Especialista v2 (19 faixas) | 5,9941 | 5,9903 ± 0,0413 | **0,0039** | 0,7655 | **2,0 KiB** | 0,0053 ms/amostra |
| Random Forest | 6,1402 | 6,1172 ± 0,0407 | 1,5487 | 0,7569 | 109,4 MiB | 0,0160 ms/amostra |
| Árvore de Decisão | 6,2824 | 6,3585 ± 0,0355 | 0,4545 | 0,7429 | 115 KiB | **0,0014 ms/amostra** |
| Especialista v1 (baseline) | 11,2721 | — | 0,0263 | 0,1964 | — | — |

O Gradient Boosting é o melhor em MAE e R², com gap pequeno. O especialista v2
fica muito próximo, é o menor artefato e o mais estável, mas exige manutenção
manual. A v1 é claramente inadequada e serve apenas como linha de base.

### Recomendação

- **Classificação:** Gradient Boosting como modelo principal; árvore de decisão
  como alternativa interpretável/edge; especialista como camada de triagem e
  auditoria (com a ressalva da cobertura).
- **Regressão:** Gradient Boosting como modelo principal; especialista v2 quando
  a explicabilidade/auditoria for obrigatória.

## Relatório final

O relatório técnico completo da atividade está em
[`docs/relatorio_final.pdf`](docs/relatorio_final.pdf) (17 páginas), com versão
HTML em [`docs/relatorio_final.html`](docs/relatorio_final.html). Ele cobre
introdução, protocolo experimental, sistema especialista, árvore, ensembles,
resultados, generalização, eficiência/energia, respostas às perguntas (a)–(f),
limitações, conclusões e referências. Os campos de identificação e os links de
dataset/repositório aparecem entre colchetes e devem ser preenchidos antes da
entrega.

## Sistema especialista e cronometragem

A base de classificação tem **4 regras** e usa apenas 3 variáveis (glicemia,
HbA1c e idade). A base de regressão v2 tem **19 faixas aditivas** com pontuação
e usa 8 variáveis. Os detalhes e os arquivos legíveis estão em `rules/`.

O esforço humano registrado é:

| Problema | Sessões | Regras cronometradas | Tempo total | Situação |
|---|---:|---:|---:|---|
| Classificação | 3 | 8 (consolidadas em 4) | 152,45 min | Completo |
| Regressão | 1 | 4 (v1) | 68,22 min | **Parcial** |

> **Atenção:** a cronometragem da regressão está incompleta (1 de 3 sessões) e a
> versão v2 (19 faixas) não foi cronometrada. O valor de 68,22 min é um limite
> inferior e se refere à v1. Consulte [`logs/README.md`](logs/README.md) para o
> formato e para as sessões pendentes.

## Limitações

- Uma única divisão treino/teste; a CV de 5 folds reduz, mas não elimina, o
  risco de conclusões instáveis.
- Classe `Low` com menos de 1% do teste; métricas macro são sensíveis a poucos
  casos.
- Regras elaboradas manualmente, com limiares clínicos e calibração OLS no
  treino; não houve validação clínica externa.
- Tempo de ML medido apenas no refit final; a busca de hiperparâmetros não foi
  cronometrada, então o custo de construção dos modelos é maior que o reportado.
- Energia e CO₂ são estimativas (`P × t`, 65 W e 45,8 gCO₂/kWh), não medições
  de um medidor de parede.
- Importância de features calculada por impureza; não foram usados SHAP ou
  permutation importance.
- Resultados dependem do hardware e da versão das bibliotecas.

## Contato

[preencher e-mail institucional, se desejado]
