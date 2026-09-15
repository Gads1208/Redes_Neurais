"""
Módulo de Construção de Pipelines de Machine Learning.
Implementa pré-processamento via ColumnTransformer sem data leakage
e encapsulamento de log-transform na target via TransformedTargetRegressor.
"""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    HIGH_CARDINALITY_FEATURES,
    NUMERICAL_FEATURES,
)


def create_preprocessor() -> ColumnTransformer:
    """
    Cria o ColumnTransformer responsável pelo pré-processamento de todos os tipos de features.
    Garante que imputações e escalonamentos sejam ajustados estritamente nos dados de treino.
    """
    # 1. Pipeline numérico: Imputação pela mediana + Padronização com StandardScaler
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 2. Pipeline categórico regular: Imputação por constante + One-Hot Encoding
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 3. Pipeline de alta cardinalidade (brand, model):
    # Agrupa categorias raras além do top-35 para evitar explosão dimensional e overfitting
    high_card_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido")),
            (
                "ohe_grouped",
                OneHotEncoder(
                    max_categories=35,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("high_card", high_card_pipeline, HIGH_CARDINALITY_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def build_regression_pipeline(
    regressor, use_log_target: bool = True
) -> TransformedTargetRegressor | Pipeline:
    """
    Constrói a pipeline completa unindo pré-processamento e o estimador de regressão.
    Se use_log_target=True, encapsula em TransformedTargetRegressor com log1p/expm1.
    """
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("regressor", regressor),
        ]
    )

    if use_log_target:
        return TransformedTargetRegressor(
            regressor=full_pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    return full_pipeline
