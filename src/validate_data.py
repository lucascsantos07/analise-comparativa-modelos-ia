
from pathlib import Path

import pandas as pd

import numpy as np


# Caminho raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# Arquivos de entrada
FILES = {
    "regression": DATA_DIR / "dataset_regression.csv",
    "classification": DATA_DIR / "dataset_classification.csv",
}

# Features e targets definidos na atividade
REGRESSION_FEATURES = [
    "Fasting_Blood_Sugar",
    "HbA1c",
    "Age",
    "BMI",
    "Family_History_Diabetes",
    "Hypertension",
    "Diet_Quality",
    "Physical_Activity_Level",
]

CLASSIFICATION_FEATURES = [
    "Fasting_Blood_Sugar",
    "HbA1c",
    "Age",
    "BMI",
    "Family_History_Diabetes",
    "Hypertension",
    "Diet_Quality",
    "PCOS",
    "Heart_Disease",
]

TARGETS = {
    "regression": "Diabetes_Risk_Score",
    "classification": "Diabetes_Risk",
}

FEATURES = {
    "regression": REGRESSION_FEATURES,
    "classification": CLASSIFICATION_FEATURES,
}


def validate_dataset(problem, file_path):
    """Verifica a integridade básica de um CSV sem modificá-lo."""

    print("\n" + "=" * 65)
    print(f"VALIDAÇÃO: {problem.upper()}")
    print("=" * 65)

    if not file_path.exists():
        print(f"ERRO: arquivo não encontrado: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as error:
        print(f"ERRO ao ler o CSV: {error}")
        return

    target = TARGETS[problem]
    expected_features = FEATURES[problem]

    print(f"Arquivo: {file_path.name}")
    print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"Colunas duplicadas: {df.columns.duplicated().sum()}")
    print(f"Linhas totalmente duplicadas: {df.duplicated().sum()}")
    print(f"Total de valores ausentes: {df.isna().sum().sum()}")

    # Verifica as colunas obrigatórias
    required_columns = expected_features + [target]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    print("\n--- Compatibilidade das colunas ---")
    if missing_columns:
        print(f"COLUNAS AUSENTES: {missing_columns}")
    else:
        print("OK: todas as features e o target estão presentes.")

    # Exibe os tipos das colunas relevantes
    available_columns = [
        col for col in required_columns if col in df.columns
    ]

    print("\n--- Tipos de dados ---")
    print(df[available_columns].dtypes.to_string())

    # Verifica valores ausentes nas variáveis utilizadas
    if available_columns:
        missing_by_column = df[available_columns].isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]

        print("\n--- Ausências nas variáveis utilizadas ---")
        if missing_by_column.empty:
            print("Nenhum valor ausente nas colunas utilizadas.")
        else:
            print(missing_by_column.to_string())

    # Verifica o target de regressão
    if problem == "regression" and target in df.columns:
        print("\n--- Target de regressão ---")
        print(f"Tipo numérico: {pd.api.types.is_numeric_dtype(df[target])}")
        print(f"Valores ausentes: {df[target].isna().sum()}")
        print(f"Valores únicos: {df[target].nunique()}")

        if pd.api.types.is_numeric_dtype(df[target]):
            print(
                f"Valores infinitos: {np.isinf(df[target]).sum()}"
            )

            print("\nEstatísticas descritivas do target:")
            print(df[target].describe().to_string())

    # Verifica o target de classificação
    if problem == "classification" and target in df.columns:
        print("\n--- Target de classificação ---")
        print(f"Valores ausentes: {df[target].isna().sum()}")
        print(f"Número de classes: {df[target].nunique()}")
        print("\nFrequência das classes:")
        print(df[target].value_counts(dropna=False).to_string())
        print("\nProporção das classes (%):")
        print(
            (df[target].value_counts(normalize=True, dropna=False) * 100)
            .round(2)
            .to_string()
        )

    # Alerta sobre tipos não numéricos nas features
    existing_features = [
        col for col in expected_features if col in df.columns
    ]

    if existing_features:
        non_numeric = df[existing_features].select_dtypes(
            exclude="number"
        ).columns.tolist()

        print("\n--- Features não numéricas ---")
        if non_numeric:
            print(
                "Verificar compatibilidade com o modelo:",
                non_numeric,
            )
        else:
            print("Todas as features são numéricas.")

    print("\nValidação concluída. Nenhum dado foi alterado.")


def main():
    for problem, file_path in FILES.items():
        validate_dataset(problem, file_path)


if __name__ == "__main__":
    main()
