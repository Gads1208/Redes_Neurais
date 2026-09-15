"""
Pipeline Robusto de Machine Learning para Previsão de Preços de Veículos.
Totalmente livre de Data Leakage, com Engenharia de Features e Transformação Logarítmica.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import ALL_FEATURE_COLUMNS, clean_raw_data, engineer_features
from src.models import calculate_metrics, get_model_candidates
from src.pipeline import build_regression_pipeline

plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (12, 8)


class CarPricePredictor:
    """
    Orquestrador completo de Machine Learning para precificação de veículos usados.
    Encapsula limpeza, engenharia de features, treinamento e inferência.
    """

    def __init__(self, use_log_target: bool = True):
        self.use_log_target = use_log_target
        self.pipelines: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_pipeline: Optional[Any] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def load_and_preprocess_data(
        self, file_path: str | Path, sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Carrega os dados brutos, aplica filtros de sanidade e engenharia de features.
        """
        path = Path(file_path)
        print(f"📊 Carregando dados de: {path.name}...")

        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        raw_df = pd.read_csv(path, encoding="ISO-8859-1")
        print(f"✓ Dados brutos carregados: {raw_df.shape[0]:,} linhas, {raw_df.shape[1]} colunas")

        # Limpeza inicial de regras de negócio (sem vazamento estatístico)
        cleaned_df = clean_raw_data(raw_df)
        print(f"✓ Registros válidos após filtros de consistência: {cleaned_df.shape[0]:,}")

        # Engenharia de atributos
        processed_df = engineer_features(cleaned_df)
        print(f"✓ Engenharia de features aplicada: {len(ALL_FEATURE_COLUMNS)} features preparadas")

        if sample_size and sample_size < len(processed_df):
            processed_df = processed_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"✓ Amostragem aplicada para treino ágil: {len(processed_df):,} registros")

        return processed_df

    def prepare_and_split_data(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ):
        """
        Separa as variáveis explicativas e a target e realiza a divisão treino/teste.
        """
        if "price" not in df.columns:
            raise KeyError("A coluna 'price' não está presente no DataFrame.")

        y = df["price"].values
        X = df[ALL_FEATURE_COLUMNS].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(
            f"\n🔄 Dados divididos com sucesso: {len(self.X_train):,} treino | {len(self.X_test):,} teste"
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """
        Treina cada modelo candidato utilizando a pipeline com ColumnTransformer.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Os dados de treino ainda não foram preparados. Execute prepare_and_split_data.")

        print("\n🚀 Iniciando treinamento do benchmark de modelos...")
        candidates = get_model_candidates()

        for name, estimator in candidates.items():
            print(f"  ⏳ Treinando {name}...")
            pipe = build_regression_pipeline(estimator, use_log_target=self.use_log_target)
            pipe.fit(self.X_train, self.y_train)
            self.pipelines[name] = pipe
            print(f"  ✓ {name} finalizado.")

        print("✅ Treinamento de todos os modelos concluído!")

    def evaluate_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Avalia o desempenho de todos os modelos no conjunto de teste independente.
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Conjunto de teste não disponível.")

        print("\n📊 Avaliando desempenho nos dados de teste...")
        self.results = {}
        best_r2 = -float("inf")

        for name, pipe in self.pipelines.items():
            y_pred = pipe.predict(self.X_test)
            metrics = calculate_metrics(self.y_test, y_pred)
            metrics["predictions"] = y_pred
            self.results[name] = metrics

            if metrics["R²"] > best_r2:
                best_r2 = metrics["R²"]
                self.best_model_name = name
                self.best_pipeline = pipe

        # Exibição organizada dos resultados
        print("\n" + "=" * 76)
        print(f"{'Modelo':<26} {'MAE (€)':<14} {'RMSE (€)':<14} {'R²':<10} {'MAPE (%)':<10}")
        print("=" * 76)

        for name, metrics in self.results.items():
            print(
                f"{name:<26} €{metrics['MAE']:<12,.2f} €{metrics['RMSE']:<12,.2f} "
                f"{metrics['R²']:<10.4f} {metrics['MAPE']:<10.2f}%"
            )

        print("=" * 76)
        print(f"🏆 Melhor Modelo: {self.best_model_name} (R² = {best_r2:.4f})\n")

        return self.results

    def plot_model_comparison(
        self, save_path: Optional[str | Path] = None, show: bool = False
    ):
        """
        Gera gráficos diagnósticos comparativos do benchmark e do melhor modelo.
        """
        if not self.results or not self.best_model_name:
            print("⚠️ É necessário avaliar os modelos antes de gerar os gráficos.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle("Comparação de Desempenho e Diagnóstico dos Modelos", fontsize=16, fontweight="bold")

        names = list(self.results.keys())
        mae_vals = [self.results[n]["MAE"] for n in names]
        r2_vals = [self.results[n]["R²"] for n in names]

        # 1. Comparativo MAE
        ax1 = axes[0, 0]
        bars1 = ax1.bar(names, mae_vals, color="#3498db", alpha=0.85, edgecolor="black")
        ax1.set_title("Erro Médio Absoluto (MAE) - Menor é melhor", fontsize=12)
        ax1.set_ylabel("MAE (€)")
        ax1.tick_params(axis="x", rotation=30)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"€{yval:,.0f}", ha="center", va="bottom", fontsize=9)

        # 2. Comparativo R²
        ax2 = axes[0, 1]
        bars2 = ax2.bar(names, r2_vals, color="#2ecc71", alpha=0.85, edgecolor="black")
        ax2.set_title("Coeficiente de Determinação (R²) - Maior é melhor", fontsize=12)
        ax2.set_ylabel("R² Score")
        ax2.tick_params(axis="x", rotation=30)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, max(0.01, yval + 0.02), f"{yval:.3f}", ha="center", va="bottom", fontsize=9)

        # 3. Real vs Previsto (Melhor Modelo)
        ax3 = axes[1, 0]
        y_test_arr = np.asarray(self.y_test)
        best_preds_arr = np.asarray(self.results[self.best_model_name]["predictions"])
        n_samples = len(y_test_arr)
        sample_size = min(1500, n_samples)
        sample_idx = np.random.choice(n_samples, size=sample_size, replace=False)
        y_sample = y_test_arr[sample_idx]
        preds_sample = best_preds_arr[sample_idx]

        ax3.scatter(y_sample, preds_sample, alpha=0.4, color="#9b59b6", s=18)
        max_val = min(60000, max(y_sample.max(), preds_sample.max(), 1000))
        ax3.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Predição Perfeita (y = x)")
        ax3.set_xlim(0, max_val)
        ax3.set_ylim(0, max_val)
        ax3.set_title(f"Valores Reais vs Previstos ({self.best_model_name})", fontsize=12)
        ax3.set_xlabel("Preço Real (€)")
        ax3.set_ylabel("Preço Previsto (€)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Distribuição dos Resíduos (Melhor Modelo)
        ax4 = axes[1, 1]
        residuals = y_test_arr - best_preds_arr
        # Filtrar faixa razoável para visualização nítida
        filtered_residuals = residuals[(residuals >= -10000) & (residuals <= 10000)]
        ax4.hist(filtered_residuals, bins=40, color="#e67e22", alpha=0.8, edgecolor="black")
        ax4.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Erro Zero")
        ax4.axvline(residuals.mean(), color="red", linestyle=":", linewidth=2, label=f"Média: €{residuals.mean():.1f}")
        ax4.set_title(f"Distribuição dos Resíduos ({self.best_model_name})", fontsize=12)
        ax4.set_xlabel("Erro de Previsão (€)")
        ax4.set_ylabel("Frequência")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_file, dpi=300)
            print(f"✓ Gráfico salvo com sucesso em: {save_file}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def predict_price(self, **kwargs) -> float:
        """
        Realiza predição individual para novos veículos com dados brutos.
        Garante que todas as transformações da pipeline sejam aplicadas de maneira idêntica ao treino.
        """
        if self.best_pipeline is None:
            raise ValueError("Nenhum modelo foi treinado ainda! Execute train_models() e evaluate_models().")

        # Criar DataFrame com os dados fornecidos
        input_raw = pd.DataFrame([kwargs])

        # Aplicar engenharia de features nas entradas brutas
        input_engineered = engineer_features(input_raw)

        # Inferência através da pipeline completa
        predicted = self.best_pipeline.predict(input_engineered)[0]
        return float(max(100.0, predicted))


# Execução direta de demonstração
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "autos.csv"

    predictor = CarPricePredictor(use_log_target=True)

    try:
        # Carregar e pré-processar (usando amostra para benchmark rápido no exemplo)
        data = predictor.load_and_preprocess_data(dataset_path, sample_size=30000)

        # Dividir treino e teste sem vazamento
        predictor.prepare_and_split_data(data, test_size=0.2, random_state=42)

        # Treinar benchmark de modelos
        predictor.train_models()

        # Avaliar e mostrar métricas
        predictor.evaluate_models()

        # Testar inferência individual
        print("🔮 Realizando previsão de teste individual...")
        exemplo_carro = {
            "vehicleType": "limousine",
            "yearOfRegistration": 2012,
            "gearbox": "manuell",
            "powerPS": 140,
            "model": "golf",
            "kilometer": 90000,
            "fuelType": "diesel",
            "brand": "volkswagen",
            "notRepairedDamage": "nein",
        }
        preco_estimado = predictor.predict_price(**exemplo_carro)
        print(f"💰 Preço estimado para o veículo de teste: €{preco_estimado:,.2f}")

        # Gerar gráficos diagnósticos
        chart_path = base_dir / "images" / "model_comparison.png"
        predictor.plot_model_comparison(save_path=chart_path)

        print("\n🎉 Pipeline executada com sucesso total!")

    except Exception as e:
        print(f"❌ Ocorreu um erro: {e}")
        import traceback
        traceback.print_exc()