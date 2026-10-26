"""Configuração central do projeto.

Concentra caminhos, sementes, listas de features e limiares clínicos
para que todos os notebooks usem exatamente o mesmo protocolo.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
RULES_DIR = ROOT_DIR / "rules"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "artifacts" / "models"

# ---------------------------------------------------------------------------
# Protocolo experimental
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
N_REPETICOES_INFERENCIA = 30

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
TARGET_CLASSIFICACAO = "Diabetes_Risk"
TARGET_REGRESSAO = "Diabetes_Risk_Score"

# Ordem clínica usada em relatórios e matrizes de confusão.
CLASSES = ["Low", "Moderate", "High"]
CLASSE_POSITIVA = "High"

# ---------------------------------------------------------------------------
# Features (as mesmas para todos os modelos de cada problema)
# ---------------------------------------------------------------------------
NUMERICAS = ["Fasting_Blood_Sugar", "HbA1c", "Age", "BMI"]

CATEGORICAS_CLASSIFICACAO = [
    "Family_History_Diabetes",
    "Hypertension",
    "Diet_Quality",
    "PCOS",
    "Heart_Disease",
]

CATEGORICAS_REGRESSAO = [
    "Family_History_Diabetes",
    "Hypertension",
    "Diet_Quality",
    "Physical_Activity_Level",
]

FEATURES_CLASSIFICACAO = NUMERICAS + CATEGORICAS_CLASSIFICACAO
FEATURES_REGRESSAO = NUMERICAS + CATEGORICAS_REGRESSAO

# Features usadas nas regras do sistema especialista (Seção 2 do enunciado).
FEATURES_REGRAS_CLASSIFICACAO = [
    "Fasting_Blood_Sugar",
    "HbA1c",
    "Age",
]
FEATURES_REGRAS_REGRESSAO = [
    "Fasting_Blood_Sugar",
    "HbA1c",
    "BMI",
    "Age",
    "Family_History_Diabetes",
    "Hypertension",
    "Diet_Quality",
    "Physical_Activity_Level",
]

# ---------------------------------------------------------------------------
# Limiares clínicos (ADA) usados na elaboração das regras
# ---------------------------------------------------------------------------
GLICEMIA_NORMAL = 100       # mg/dL - limite superior da glicemia normal
GLICEMIA_DIABETES = 126     # mg/dL - critério diagnóstico de diabetes
HBA1C_NORMAL = 5.7          # % - limite superior do valor normal
HBA1C_DIABETES = 6.5        # % - critério diagnóstico de diabetes

# Pontos de corte do escore de risco (apenas para leitura dos resultados).
CORTE_RISCO_BAIXO = 35
CORTE_RISCO_ALTO = 65

# ---------------------------------------------------------------------------
# Estimativa de energia e carbono
# ---------------------------------------------------------------------------
POTENCIA_WATTS = 65.0           # potência média estimada do processador
FATOR_CO2_BRASIL = 45.8         # gCO2/kWh (fator apresentado na disciplina)


def criar_diretorios() -> None:
    """Garante que as pastas de saída existam."""
    for diretorio in (TABLES_DIR, FIGURES_DIR, MODELS_DIR):
        diretorio.mkdir(parents=True, exist_ok=True)
