"""
Módulo de Modelos e Avaliação.
Disponibiliza os estimadores para benchmark comparativo e cálculo de métricas de regressão.
"""

from typing import Any, Dict
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


def get_model_candidates() -> Dict[str, Any]:
    """
    Retorna o portfólio de modelos para benchmark comparativo:
    - Baseline (DummyRegressor pela mediana)
    - Modelo Linear Regularizado (Ridge)
    - Random Forest (Ensemble baseado em Bagging)
    - HistGradientBoosting (Ensemble baseado em Boosting de alta velocidade)
    - Rede Neural MLP (Multi-Layer Perceptron com Early Stopping)
    """
    return {
        "Baseline (Mediana)": DummyRegressor(strategy="median"),
        "Regressão Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_split=8,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=8,
            learning_rate=0.08,
            random_state=42,
        ),
        "Rede Neural (MLP)": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        ),
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula as principais métricas de regressão financeira e imobiliária/veicular:
    - MAE: Erro Médio Absoluto (em Euros)
    - RMSE: Raiz do Erro Quadrático Médio
    - R²: Coeficiente de Determinação (variância explicada)
    - MAPE: Erro Percentual Absoluto Médio (%)
    """
    # Garantir que previsões sejam não-negativas
    y_pred_clipped = np.clip(y_pred, a_min=1.0, a_max=None)
    y_true_clipped = np.clip(y_true, a_min=1.0, a_max=None)

    mae = mean_absolute_error(y_true, y_pred_clipped)
    mse = mean_squared_error(y_true, y_pred_clipped)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred_clipped)

    # MAPE seguro contra divisão por zero
    mape = float(np.mean(np.abs((y_true_clipped - y_pred_clipped) / y_true_clipped)) * 100)

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(rmse, 2),
        "R²": round(float(r2), 4),
        "MAPE": round(mape, 2),
    }
