# Versão alternativa usando apenas Scikit-learn
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

class CarPricePredictorML:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_and_preprocess_data(self, file_path):
        """Carrega e pré-processa os dados"""
        print("📊 Carregando dados...")
        
        try:
            self.base = pd.read_csv(file_path, encoding="ISO-8859-1")
            print(f"✓ Dados carregados: {self.base.shape[0]} registros, {self.base.shape[1]} colunas")
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return None
            
        # Análise exploratória inicial
        self._exploratory_analysis()
        
        # Remoção de atributos irrelevantes
        lista_atributos = ["dateCrawled", "seller", "offerType", 
                          "nrOfPictures", "postalCode", "lastSeen", 
                          "dateCreated", "name"]
        self.base = self.base.drop(lista_atributos, axis=1, errors='ignore')
        
        # Filtros na base
        initial_count = len(self.base)
        
        # Filtros de preço baseados em quartis
        Q1 = self.base.price.quantile(0.01)
        Q3 = self.base.price.quantile(0.99)
        self.base = self.base[(self.base.price >= Q1) & (self.base.price <= Q3)]
        
        # Filtros adicionais
        self.base = self.base[self.base.price > 100]
        if 'yearOfRegistration' in self.base.columns:
            self.base = self.base[(self.base.yearOfRegistration >= 1950) & 
                                (self.base.yearOfRegistration <= 2024)]
        if 'powerPS' in self.base.columns:
            self.base = self.base[(self.base.powerPS >= 50) & (self.base.powerPS <= 1000)]
        if 'kilometer' in self.base.columns:
            self.base = self.base[self.base.kilometer <= 500000]
            
        print(f"✓ Filtros aplicados: {initial_count - len(self.base)} registros removidos")
        
        return self.base
    
    def _exploratory_analysis(self):
        """Análise exploratória dos dados"""
        print("\n📈 Análise Exploratória dos Dados")
        print("=" * 50)
        
        print(f"Dimensões: {self.base.shape}")
        print(f"\nValores ausentes por coluna:")
        missing_data = self.base.isnull().sum()
        missing_percentage = (missing_data / len(self.base)) * 100
        missing_df = pd.DataFrame({
            'Ausentes': missing_data,
            'Percentual': missing_percentage
        }).sort_values('Percentual', ascending=False)
        print(missing_df[missing_df.Ausentes > 0])
        
        print(f"\n📊 Estatísticas do Preço:")
        print(self.base.price.describe())
        
    def handle_missing_values(self):
        """Tratamento de valores faltosos"""
        print("\n🔧 Tratando valores ausentes...")
        
        for column in self.base.columns:
            if self.base[column].isnull().sum() > 0:
                if self.base[column].dtype == 'object':
                    if self.base[column].isnull().sum() / len(self.base) > 0.3:
                        self.base[column] = self.base[column].fillna('unknown')
                    else:
                        mode_value = self.base[column].mode()
                        if len(mode_value) > 0:
                            self.base[column] = self.base[column].fillna(mode_value[0])
                else:
                    median_value = self.base[column].median()
                    self.base[column] = self.base[column].fillna(median_value)
        
        print("✓ Valores ausentes tratados")
        
    def prepare_features(self):
        """Preparação das features"""
        print("\n⚙️ Preparando features...")
        
        if 'price' not in self.base.columns:
            raise ValueError("Coluna 'price' não encontrada!")
            
        y = self.base['price'].values
        X = self.base.drop('price', axis=1)
        
        # Encoding de variáveis categóricas
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        self.feature_names = X.columns.tolist()
        print(f"✓ Features preparadas: {len(self.feature_names)} variáveis")
        
        return X.values, y
    
    def train_models(self, X_train, y_train):
        """Treina múltiplos modelos"""
        print("\n🚀 Treinando modelos...")
        
        # Definir modelos
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'Neural Network (MLP)': MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        # Treinar cada modelo
        for name, model in models.items():
            print(f"🔄 Treinando {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"✓ {name} treinado")
        
        print("✅ Todos os modelos treinados!")
    
    def evaluate_models(self, X_test, y_test):
        """Avalia todos os modelos"""
        print("\n📊 Avaliando modelos...")
        
        results = {}
        best_r2 = -float('inf')
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
                'predictions': y_pred
            }
            
            # Encontrar melhor modelo
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.best_model_name = name
        
        # Mostrar resultados
        print("\n🏆 RESULTADOS DOS MODELOS")
        print("=" * 80)
        print(f"{'Modelo':<20} {'MAE':<12} {'RMSE':<12} {'R²':<8} {'MAPE':<8}")
        print("-" * 80)
        
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['MAE']:<12.2f} {metrics['RMSE']:<12.2f} "
                  f"{metrics['R²']:<8.4f} {metrics['MAPE']:<8.2f}%")
        
        print(f"\n🥇 Melhor modelo: {self.best_model_name} (R² = {best_r2:.4f})")
        
        return results
    
    def plot_model_comparison(self, results, y_test):
        """Plota comparação entre modelos"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparação de Modelos', fontsize=16, fontweight='bold')
        
        # Preparar dados
        model_names = list(results.keys())
        metrics = ['MAE', 'RMSE', 'R²', 'MAPE']
        
        # 1. Comparação de métricas
        ax1 = axes[0, 0]
        mae_values = [results[name]['MAE'] for name in model_names]
        bars = ax1.bar(model_names, mae_values, alpha=0.7)
        ax1.set_title('MAE por Modelo')
        ax1.set_ylabel('MAE')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. R² Score
        ax2 = axes[0, 1]
        r2_values = [results[name]['R²'] for name in model_names]
        bars = ax2.bar(model_names, r2_values, alpha=0.7, color='green')
        ax2.set_title('R² Score por Modelo')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Real vs Previsto (melhor modelo)
        ax3 = axes[1, 0]
        best_pred = results[self.best_model_name]['predictions']
        ax3.scatter(y_test, best_pred, alpha=0.6, s=20)
        min_val = min(y_test.min(), best_pred.min())
        max_val = max(y_test.max(), best_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax3.set_xlabel('Valores Reais')
        ax3.set_ylabel('Valores Previstos')
        ax3.set_title(f'Real vs Previsto - {self.best_model_name}')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribuição dos erros (melhor modelo)
        ax4 = axes[1, 1]
        errors = y_test - best_pred
        ax4.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(errors.mean(), color='red', linestyle='--', 
                   label=f'Média: {errors.mean():.2f}')
        ax4.set_xlabel('Resíduos')
        ax4.set_ylabel('Frequência')
        ax4.set_title(f'Distribuição dos Resíduos - {self.best_model_name}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, X_test, y_test, n_features=10):
        """Plota importância das features"""
        print(f"\n🔍 Calculando importância das features...")
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Para modelos baseados em árvore
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:n_features]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(indices)), importances[indices])
            plt.xlabel('Features')
            plt.ylabel('Importância')
            plt.title(f'Importância das Features - {self.best_model_name}')
            plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            # Para outros modelos, usar permutation importance
            perm_importance = permutation_importance(
                self.best_model, X_test, y_test, n_repeats=5, random_state=42
            )
            
            # Ordenar por importância
            sorted_idx = perm_importance.importances_mean.argsort()[::-1][:n_features]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
            plt.xlabel('Features')
            plt.ylabel('Redução no Score')
            plt.title(f'Importância das Features (Permutation) - {self.best_model_name}')
            plt.xticks(range(len(sorted_idx)), 
                      [self.feature_names[i] for i in sorted_idx], rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def predict_price(self, **kwargs):
        """Faz previsão individual"""
        if self.best_model is None:
            raise ValueError("Nenhum modelo foi treinado ainda!")
        
        # Criar DataFrame
        input_data = pd.DataFrame([kwargs])
        
        # Aplicar transformações
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                try:
                    input_data[col] = self.label_encoders[col].transform(input_data[col])
                except ValueError:
                    print(f"Aviso: Valor desconhecido para {col}, usando 0")
                    input_data[col] = 0
        
        # Normalizar
        input_scaled = self.scaler.transform(input_data.values)
        
        # Prever
        price_pred = self.best_model.predict(input_scaled)[0]
        
        return price_pred

# Exemplo de uso
if __name__ == "__main__":
    predictor = CarPricePredictorML()
    
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "autos.csv"
    
    try:
        # Pipeline completo
        data = predictor.load_and_preprocess_data(file_path)
        if data is None:
            exit()
        
        predictor.handle_missing_values()
        X, y = predictor.prepare_features()
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalizar
        X_train_scaled = predictor.scaler.fit_transform(X_train)
        X_test_scaled = predictor.scaler.transform(X_test)
        
        # Treinar modelos
        predictor.train_models(X_train_scaled, y_train)
        
        # Avaliar
        results = predictor.evaluate_models(X_test_scaled, y_test)
        
        # Visualizações
        predictor.plot_model_comparison(results, y_test)
        predictor.plot_feature_importance(X_test_scaled, y_test)
        
        print("\n🎉 Análise completa finalizada!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()