"""
Módulo de Limpeza e Engenharia de Features.
Implementa transformações consistentes e elimina riscos de data leakage.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

# Listas de colunas categorizadas por tipo de transformação
NUMERICAL_FEATURES: List[str] = [
    "vehicle_age",
    "powerPS",
    "kilometer",
    "km_per_year",
]

CATEGORICAL_FEATURES: List[str] = [
    "vehicleType",
    "gearbox",
    "fuelType",
    "notRepairedDamage",
]

HIGH_CARDINALITY_FEATURES: List[str] = [
    "brand",
    "model",
]

BINARY_FEATURES: List[str] = [
    "is_damaged",
    "is_premium",
]

ALL_FEATURE_COLUMNS = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES + HIGH_CARDINALITY_FEATURES + BINARY_FEATURES
)

PREMIUM_BRANDS = {
    "audi",
    "bmw",
    "mercedes_benz",
    "porsche",
    "jaguar",
    "land_rover",
    "volvo",
}


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza preliminar de registros inválidos e inconsistentes do dataset.
    Aplica regras de negócio para remoção de ruídos extremos e anúncios fraudulentos.
    """
    data = df.copy()

    # Remover atributos irrelevantes ou de metadados da coleta
    drop_columns = [
        "dateCrawled",
        "seller",
        "offerType",
        "nrOfPictures",
        "postalCode",
        "lastSeen",
        "dateCreated",
        "name",
        "abtest",
    ]
    data = data.drop(columns=[c for c in drop_columns if c in data.columns], errors="ignore")

    # Filtros de sanidade para o preço (variável target)
    if "price" in data.columns:
        # Preços simbólicos (ex: €0 ou €1) e preços astronômicos irreais
        data = data[(data["price"] >= 200) & (data["price"] <= 120000)]

    # Filtros de sanidade para ano de registro (dataset coletado em 2016)
    if "yearOfRegistration" in data.columns:
        data = data[
            (data["yearOfRegistration"] >= 1960) & (data["yearOfRegistration"] <= 2016)
        ]

    # Filtros de sanidade para potência do motor (em cavalos / PS)
    if "powerPS" in data.columns:
        data = data[(data["powerPS"] >= 30) & (data["powerPS"] <= 700)]

    # Filtros de quilometragem
    if "kilometer" in data.columns:
        data = data[(data["kilometer"] >= 5000) & (data["kilometer"] <= 250000)]

    return data.reset_index(drop=True)


def engineer_features(df: pd.DataFrame, reference_year: int = 2016) -> pd.DataFrame:
    """
    Cria novas variáveis informativas a partir dos atributos brutos existentes:
    - vehicle_age: idade calculada do veículo.
    - km_per_year: média anual de quilometragem rodada.
    - is_damaged: se o carro possui avarias registradas.
    - is_premium: categorização de marcas de luxo/prestígio.
    """
    data = df.copy()

    # 1. Idade do veículo em relação ao ano do crawl (2016)
    year_reg = data["yearOfRegistration"] if "yearOfRegistration" in data.columns else reference_year
    vehicle_age = np.clip(reference_year - year_reg, 0, 60)
    data["vehicle_age"] = vehicle_age

    # 2. Quilometragem por ano (indicador de intensidade de uso)
    kilometer = data["kilometer"] if "kilometer" in data.columns else 100000
    data["km_per_year"] = kilometer / (data["vehicle_age"] + 1)

    # 3. Indicador binário de avaria não reparada
    if "notRepairedDamage" in data.columns:
        data["is_damaged"] = (data["notRepairedDamage"].astype(str).str.lower() == "ja").astype(int)
        # Padronizar faltantes
        data["notRepairedDamage"] = data["notRepairedDamage"].fillna("desconhecido")
    else:
        data["is_damaged"] = 0
        data["notRepairedDamage"] = "desconhecido"

    # 4. Indicador de marca premium
    if "brand" in data.columns:
        data["is_premium"] = (
            data["brand"].astype(str).str.lower().isin(PREMIUM_BRANDS)
        ).astype(int)
        data["brand"] = data["brand"].fillna("desconhecido")
    else:
        data["is_premium"] = 0
        data["brand"] = "desconhecido"

    # 5. Preenchimento seguro de nulos em colunas categóricas para evitar quebras
    for col in ["vehicleType", "gearbox", "fuelType", "model"]:
        if col in data.columns:
            data[col] = data[col].fillna("desconhecido").astype(str)
        else:
            data[col] = "desconhecido"

    # Garantir que todas as colunas esperadas estejam presentes no DataFrame
    for col in ALL_FEATURE_COLUMNS:
        if col not in data.columns:
            if col in NUMERICAL_FEATURES:
                data[col] = 0.0
            elif col in BINARY_FEATURES:
                data[col] = 0
            else:
                data[col] = "desconhecido"

    return data
