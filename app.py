"""
Aplicação Web Interativa em Streamlit para Demonstração de Portfólio.
Previsão Inteligente de Preços de Veículos Usados com Machine Learning e Redes Neurais.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import ALL_FEATURE_COLUMNS, clean_raw_data, engineer_features
from src.pipeline import build_regression_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Car Price Predictor | Deep Learning & ML",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS personalizada para acabamento de portfólio de alto nível
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        border: 1px solid #edf2f7;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #718096;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #2b6cb0;
        margin-top: 4px;
    }
    .price-box {
        background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
        color: #ffffff;
        padding: 26px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(43, 108, 176, 0.25);
        text-align: center;
        margin: 20px 0;
    }
    .price-box h2 {
        color: #ffffff !important;
        font-size: 2.6rem;
        margin: 8px 0;
        font-weight: 800;
    }
    .price-sub {
        font-size: 1.1rem;
        color: #e2e8f0;
    }
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background-color: #ebf8ff; color: #2b6cb0; }
    .badge-green { background-color: #f0fff4; color: #276749; }
    .badge-amber { background-color: #fffaf0; color: #c05621; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_pipeline.joblib"
DATASET_PATH = BASE_DIR / "autos.csv"


@st.cache_resource(show_spinner="Carregando modelo inteligente de precificação...")
def get_prediction_pipeline():
    """
    Carrega o modelo serializado do disco ou treina uma versão compacta
    no primeiro acesso caso o arquivo ainda não exista.
    """
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # Caso ainda não exista, treina um pipeline de alta performance
    if not DATASET_PATH.exists():
        st.error(f"Dataset não encontrado em: {DATASET_PATH}")
        return None

    raw_df = pd.read_csv(DATASET_PATH, encoding="ISO-8859-1", nrows=35000)
    clean_df = clean_raw_data(raw_df)
    features_df = engineer_features(clean_df)

    y = features_df["price"].values
    X = features_df[ALL_FEATURE_COLUMNS]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.15, random_state=42)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        alpha=0.01,
        max_iter=250,
        random_state=42,
        early_stopping=True,
    )
    pipeline = build_regression_pipeline(mlp, use_log_target=True)
    pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


# Carregamento do pipeline
pipeline = get_prediction_pipeline()

# ==================== BARRA LATERAL (SIDEBAR) ====================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/car--v1.png", width=70)
    st.markdown("### 🚗 Car Price AI")
    st.caption("Sistema Preditivo com Redes Neurais & MLOps")

    st.markdown("---")
    st.markdown("#### ⚡ Carregar Presets Rápidos")
    preset = st.selectbox(
        "Selecione um veículo de teste:",
        [
            "Personalizado (Manual)",
            "VW Golf 2.0 TDI (2012)",
            "BMW 320i Sedan (2014)",
            "Audi A4 2.0 TFSI (2015)",
            "Mercedes-Benz C200 (2013)",
            "Ford Focus 1.6 (2010)",
        ],
    )

    st.markdown("---")
    st.markdown("#### 💱 Câmbio Monetário")
    eur_to_brl = st.number_input("Cotação EUR / BRL (R$):", min_value=1.0, max_value=15.0, value=6.00, step=0.1)

    st.markdown("---")
    st.markdown("#### ℹ️ Sobre o Dataset")
    st.markdown(
        """
        - **Origem:** eBay Kleinanzeigen
        - **Registros:** 370.000+ anúncios
        - **Target:** Preço real de venda (€)
        - **Métrica Campeã:** $R^2 = 0.8359$
        """
    )


# Dados padrão dos Presets
default_values = {
    "brand": "volkswagen",
    "model": "golf",
    "vehicleType": "limousine",
    "gearbox": "manuell",
    "fuelType": "diesel",
    "year": 2012,
    "kilometer": 100000,
    "power": 140,
    "damage": "Não",
}

if preset == "VW Golf 2.0 TDI (2012)":
    default_values = {
        "brand": "volkswagen",
        "model": "golf",
        "vehicleType": "limousine",
        "gearbox": "manuell",
        "fuelType": "diesel",
        "year": 2012,
        "kilometer": 95000,
        "power": 140,
        "damage": "Não",
    }
elif preset == "BMW 320i Sedan (2014)":
    default_values = {
        "brand": "bmw",
        "model": "3er",
        "vehicleType": "limousine",
        "gearbox": "automatik",
        "fuelType": "benzin",
        "year": 2014,
        "kilometer": 80000,
        "power": 184,
        "damage": "Não",
    }
elif preset == "Audi A4 2.0 TFSI (2015)":
    default_values = {
        "brand": "audi",
        "model": "a4",
        "vehicleType": "kombi",
        "gearbox": "automatik",
        "fuelType": "benzin",
        "year": 2015,
        "kilometer": 75000,
        "power": 190,
        "damage": "Não",
    }
elif preset == "Mercedes-Benz C200 (2013)":
    default_values = {
        "brand": "mercedes_benz",
        "model": "c_klasse",
        "vehicleType": "limousine",
        "gearbox": "automatik",
        "fuelType": "diesel",
        "year": 2013,
        "kilometer": 110000,
        "power": 170,
        "damage": "Não",
    }
elif preset == "Ford Focus 1.6 (2010)":
    default_values = {
        "brand": "ford",
        "model": "focus",
        "vehicleType": "kleinwagen",
        "gearbox": "manuell",
        "fuelType": "benzin",
        "year": 2010,
        "kilometer": 140000,
        "power": 105,
        "damage": "Não",
    }

# ==================== CABEÇALHO DA PÁGINA ====================
st.markdown('<div class="main-header">🚗 Previsão de Preços de Veículos com Redes Neurais</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Solução ponta a ponta de Machine Learning: Engenharia de Features, eliminação de Data Leakage e inferência em tempo real.</div>',
    unsafe_allow_html=True,
)

# Abas Principais
tab_sim, tab_benchmark, tab_insights, tab_about = st.tabs(
    ["🔮 Simulador de Preço", "📊 Benchmark dos Modelos", "📈 Insights de Mercado", "👨‍💻 Sobre o Projeto"]
)

# ==================== ABA 1: SIMULADOR ====================
with tab_sim:
    st.markdown("### Preencha as características do veículo para obter a estimativa de mercado")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🚘 Identificação do Veículo")
        brands_list = [
            "volkswagen", "bmw", "mercedes_benz", "audi", "opel", "ford", "renault",
            "peugeot", "fiat", "seat", "skoda", "toyota", "nissan", "hyundai", "honda", "porsche"
        ]
        brand = st.selectbox(
            "Marca do Veículo:",
            brands_list,
            index=brands_list.index(default_values["brand"]) if default_values["brand"] in brands_list else 0,
        )

        model_name = st.text_input("Modelo (ex: golf, 3er, a4, focus, civic):", value=default_values["model"])

        car_types = {
            "limousine": "Sedan (Limousine)",
            "kleinwagen": "Hatch / Compacto (Kleinwagen)",
            "kombi": "Perua / Station Wagon (Kombi)",
            "suv": "SUV",
            "coupe": "Coupé",
            "cabrio": "Conversível (Cabrio)",
            "bus": "Van / Minivan (Bus)",
        }
        car_type_keys = list(car_types.keys())
        vehicle_type = st.selectbox(
            "Tipo de Carroceria:",
            car_type_keys,
            format_func=lambda x: car_types[x],
            index=car_type_keys.index(default_values["vehicleType"]) if default_values["vehicleType"] in car_type_keys else 0,
        )

        gearboxes = {"manuell": "Manual", "automatik": "Automático"}
        gear_keys = list(gearboxes.keys())
        gearbox = st.selectbox(
            "Tipo de Câmbio:",
            gear_keys,
            format_func=lambda x: gearboxes[x],
            index=gear_keys.index(default_values["gearbox"]) if default_values["gearbox"] in gear_keys else 0,
        )

    with col2:
        st.markdown("##### ⚙️ Estado, Quilometragem e Desempenho")
        fuel_types = {
            "benzin": "Gasolina (Benzin)",
            "diesel": "Diesel",
            "lpg": "Gás GLP (LPG)",
            "cng": "Gás Natural (CNG)",
            "hybrid": "Híbrido",
            "elektro": "Elétrico",
        }
        fuel_keys = list(fuel_types.keys())
        fuel_type = st.selectbox(
            "Combustível:",
            fuel_keys,
            format_func=lambda x: fuel_types[x],
            index=fuel_keys.index(default_values["fuelType"]) if default_values["fuelType"] in fuel_keys else 0,
        )

        year_reg = st.slider("Ano de Registro:", min_value=1980, max_value=2016, value=default_values["year"])
        kilometer = st.slider(
            "Quilometragem Rodada (km):", min_value=5000, max_value=250000, value=default_values["kilometer"], step=5000
        )
        power_ps = st.slider("Potência do Motor (cv / PS):", min_value=40, max_value=550, value=default_values["power"], step=5)

        has_damage = st.radio(
            "Possui histórico de danos não reparados?",
            ["Não", "Sim"],
            index=0 if default_values["damage"] == "Não" else 1,
            horizontal=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    calc_button = st.button("🚀 Calcular Preço de Mercado", type="primary", use_container_width=True)

    if calc_button:
        input_data = pd.DataFrame(
            [
                {
                    "brand": brand,
                    "model": model_name.strip().lower() if model_name else "desconhecido",
                    "vehicleType": vehicle_type,
                    "gearbox": gearbox,
                    "fuelType": fuel_type,
                    "yearOfRegistration": year_reg,
                    "kilometer": kilometer,
                    "powerPS": power_ps,
                    "notRepairedDamage": "ja" if has_damage == "Sim" else "nein",
                }
            ]
        )

        # Aplicar engenharia de features idêntica ao treino
        engineered_input = engineer_features(input_data)

        # Inferência através da pipeline
        if pipeline is not None:
            predicted_price = float(pipeline.predict(engineered_input)[0])
            predicted_price = max(200.0, predicted_price)
            price_brl = predicted_price * eur_to_brl

            # Margem de tolerância baseada no MAE do modelo (~€1.480)
            mae_margin = 1480.0
            min_price = max(150.0, predicted_price - mae_margin)
            max_price = predicted_price + mae_margin

            st.markdown(
                f"""
                <div class="price-box">
                    <div class="price-sub">Valor Estimado de Mercado</div>
                    <h2>€ {predicted_price:,.2f}</h2>
                    <div class="price-sub">Equivalente a aproximadamente <strong>R$ {price_brl:,.2f}</strong></div>
                    <div style="margin-top: 12px; font-size: 0.95rem; color: #cbd5e0;">
                        Faixa de negociação esperada: <strong>€{min_price:,.0f}</strong> a <strong>€{max_price:,.0f}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Badges de Diagnóstico do Veículo
            st.markdown("##### 📌 Análise de Características do Veículo:")
            badges_html = ""
            if engineered_input["is_premium"].iloc[0] == 1:
                badges_html += '<span class="badge badge-blue">⭐ Marca Premium / Alta Valorização</span>'
            else:
                badges_html += '<span class="badge badge-blue">🚗 Marca de Alto Volume Comercial</span>'

            if kilometer < 80000:
                badges_html += '<span class="badge badge-green">✨ Baixa Quilometragem</span>'
            elif kilometer > 150000:
                badges_html += '<span class="badge badge-amber">⚠️ Quilometragem Elevada</span>'

            if has_damage == "Sim":
                badges_html += '<span class="badge badge-amber">💥 Histórico com Avaria (Depreciação Severa)</span>'
            else:
                badges_html += '<span class="badge badge-green">🛡️ Sem Registro de Avarias</span>'

            st.markdown(badges_html, unsafe_allow_html=True)


# ==================== ABA 2: BENCHMARK ====================
with tab_benchmark:
    st.markdown("### 🏆 Comparativo Formal entre Modelos (Conjunto de Teste)")
    st.caption("Avaliação em 6.000 veículos independentes nunca vistos durante o treinamento.")

    # 4 Cards de Métricas Principais
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """<div class="metric-card"><div class="metric-title">Melhor R² Score</div><div class="metric-value">0.8359</div><small>Rede Neural MLP</small></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="metric-card"><div class="metric-title">Erro Médio (MAE)</div><div class="metric-value">€ 1.481</div><small>Desvio Absoluto Médio</small></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="metric-card"><div class="metric-title">RMSE</div><div class="metric-value">€ 3.141</div><small>Penalização de Erros Grandes</small></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """<div class="metric-card"><div class="metric-title">Base de Teste</div><div class="metric-value">6.000</div><small>Veículos Isolados</small></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabela comparativa
    benchmark_data = {
        "Modelo": [
            "Baseline (Mediana)",
            "Regressão Ridge (Linear Regularizado)",
            "Random Forest (Ensemble Bagging)",
            "HistGradientBoosting (Boosting Moderno)",
            "Rede Neural (MLP Profunda com Early Stopping)",
        ],
        "MAE (€)": ["€ 4.593,12", "€ 2.280,16", "€ 1.440,99", "€ 1.423,19", "€ 1.480,95"],
        "RMSE (€)": ["€ 8.216,27", "€ 6.709,37", "€ 3.249,81", "€ 3.155,25", "€ 3.141,61"],
        "R² Score": [-0.1223, 0.2516, 0.8244, 0.8345, 0.8359],
        "MAPE (%)": ["146,96%", "51,87%", "36,67%", "35,11%", "36,23%"],
    }
    df_benchmark = pd.DataFrame(benchmark_data)
    st.dataframe(df_benchmark, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📈 Gráficos Diagnósticos dos Modelos")
    chart_file = BASE_DIR / "images" / "model_comparison.png"
    if chart_file.exists():
        st.image(str(chart_file), caption="Comparativo de Métricas, Dispersão Real vs Previsto e Distribuição dos Resíduos", use_container_width=True)
    else:
        st.info("O gráfico comparativo pode ser gerado executando `python regressao_precos.py`.")


# ==================== ABA 3: INSIGHTS ====================
with tab_insights:
    st.markdown("### 🔍 Insights de Negócio e Dinâmica do Mercado Automotivo")

    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        st.markdown("#### 1. Fator Idade vs. Depreciação")
        st.write(
            """
            - A curva de desvalorização segue um decaimento log-normal: a maior taxa de perda de valor
              ocorre nos primeiros 4 anos de vida do veículo.
            - Modelos de marcas populares (Fiat, Renault, Opel) sofrem taxas de depreciação anual maiores
              do que marcas alemãs com forte valor de revenda (Volkswagen, Audi, BMW).
            """
        )

        st.markdown("#### 2. O Peso Crítico do Histórico de Avarias (`notRepairedDamage`)")
        st.write(
            """
            - Carros que sofreram colisões ou acidentes sem reparo sofrem um desconto médio imediato
              entre **35% e 50%** em relação a modelos com histórico limpo.
            - Criar uma flag binária explícita (`is_damaged`) aumentou significativamente a acurácia do modelo.
            """
        )

    with col_ins2:
        st.markdown("#### 3. Quilometragem por Ano (`km_per_year`)")
        st.write(
            """
            - Dois veículos com 100.000 km têm valores completamente diferentes se um tem 3 anos e o outro tem 10 anos.
            - A métrica de intensidade de uso anual captura se o carro foi utilizado para viagens em rodovias (frotas)
              ou para uso urbano moderado.
            """
        )

        st.markdown("#### 4. Por que a Transformação Logarítmica (`log1p`) foi Decisiva?")
        st.write(
            """
            - Em problemas financeiros e de preços, a distribuição de preços tem cauda longa para a direita.
            - Treinar com $y = \\log(1 + \\text{price})$ estabilizou o gradiente da Rede Neural e eliminou
              qualquer possibilidade do modelo prever valores negativos.
            """
        )


# ==================== ABA 4: SOBRE ====================
with tab_about:
    st.markdown("### 👨‍💻 Arquitetura de Engenharia de Machine Learning")
    st.markdown(
        """
        Este projeto foi reestruturado seguindo as melhores práticas de Engenharia de Dados e Machine Learning para portfólio profissional:

        - **Zero Data Leakage:** Imputação de dados faltantes, normalização com `StandardScaler` e codificação de variáveis categóricas via `OneHotEncoder` e agrupamento de categorias raras são ajustados estritamente no conjunto de treino através de um `ColumnTransformer`.
        - **Modelagem Robusta:** Comparação estruturada contra baselines e modelos baseados em árvores (`RandomForest`, `HistGradientBoosting`).
        - **Deep Learning Aplicado:** Rede Neural profunda em camadas densas com regularização L2 (`alpha=0.01`), otimizador Adam e mecanismo de parada antecipada (*Early Stopping*).
        - **Pipeline Portátil e Reutilizável:** Serialização via `joblib` e inferência desacoplada da interface visual.

        ---
        #### 🛠️ Tecnologias Utilizadas:
        - **Linguagem:** Python 3
        - **Bibliotecas:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib, Streamlit
        - **Versionamento:** Git com `.gitignore` completo e histórico higienizado
        """
    )
