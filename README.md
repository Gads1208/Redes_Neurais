# 🚗 Previsão de Preços de Carros com Redes Neurais

Este projeto implementa uma solução completa de **Machine Learning** usando **Redes Neurais** para prever preços de carros usados. O sistema inclui análise exploratória, pré-processamento avançado, treinamento otimizado e visualizações detalhadas.

## 🎯 Características Principais

- ✅ **Análise exploratória automática** dos dados
- ✅ **Pré-processamento inteligente** com tratamento de outliers
- ✅ **Arquiteturas de rede neural flexíveis** (simples, média, profunda)
- ✅ **Callbacks avançados** para otimização do treinamento
- ✅ **Métricas de avaliação completas** (MAE, MSE, RMSE, R², MAPE)
- ✅ **6 visualizações diferentes** para análise de performance
- ✅ **Importância das features** por análise de permutação
- ✅ **Interface orientada a objetos** para fácil reutilização

## 📊 Dataset

O projeto foi desenvolvido para trabalhar com o dataset de carros usados que contém informações como:

- **Preço** (variável target)
- **Marca e modelo** do veículo
- **Ano de fabricação**
- **Quilometragem**
- **Tipo de combustível**
- **Potência do motor**
- **Tipo de câmbio**
- **Outras características técnicas**

## 🛠️ Instalação

### 1. Clone o repositório
```bash
git clone <seu-repositorio>
cd car-price-prediction
```

### 2. Crie um ambiente virtual (recomendado)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

## 🚀 Como Usar

### Uso Básico

```python
from car_price_predictor import CarPricePredictorNN

# Inicializar o preditor
predictor = CarPricePredictorNN()

# Caminho para seu arquivo CSV
file_path = "caminho/para/seu/arquivo.csv"

# Pipeline completo
try:
    # 1. Carregar e pré-processar dados
    data = predictor.load_and_preprocess_data(file_path)
    
    # 2. Tratar valores ausentes
    predictor.handle_missing_values()
    
    # 3. Preparar features
    X, y = predictor.prepare_features()
    
    # 4. Dividir e normalizar dados
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_and_scale_data(X, y)
    
    # 5. Construir modelo (opções: 'simple', 'medium', 'deep')
    model = predictor.build_model(X_train.shape[1], architecture='medium')
    
    # 6. Treinar modelo
    predictor.train_model(X_train, X_val, y_train, y_val, epochs=100, batch_size=64)
    
    # 7. Avaliar modelo com gráficos
    results = predictor.evaluate_model(X_test, y_test, show_plots=True)
    
    # 8. Analisar importância das features
    predictor.plot_feature_importance_approximation(X_test, y_test)
    
    print("✅ Análise completa finalizada!")
    
except Exception as e:
    print(f"❌ Erro: {e}")
```

### Fazendo Previsões Individuais

```python
# Após treinar o modelo, você pode fazer previsões individuais
predicted_price = predictor.predict_price(
    vehicleType='sedan',
    yearOfRegistration=2015,
    gearbox='manual',
    powerPS=120,
    model='golf',
    kilometer=50000,
    fuelType='petrol',
    brand='volkswagen'
)

print(f"Preço previsto: €{predicted_price:,.2f}")
```

## 📈 Funcionalidades Avançadas

### Arquiteturas Disponíveis

1. **Simple**: Rede neural básica (64→32→1)
2. **Medium**: Arquitetura balanceada com regularização (256→128→64→32→1)
3. **Deep**: Rede profunda para dados complexos (512→256→128→64→32→16→1)

### Métricas de Avaliação

- **MAE** (Mean Absolute Error): Erro médio em valor absoluto
- **MSE** (Mean Squared Error): Erro quadrático médio
- **RMSE** (Root MSE): Raiz do erro quadrático médio
- **R²** (Coefficient of Determination): Coeficiente de determinação
- **MAPE** (Mean Absolute Percentage Error): Erro percentual médio

### Visualizações Geradas

1. 📊 **Evolução da Loss**: Acompanha o treinamento e validação
2. 🎯 **Real vs Previsto**: Scatter plot para verificar precisão
3. 📈 **Distribuição dos Resíduos**: Histograma dos erros
4. 🔍 **Resíduos vs Previsões**: Detecta padrões nos erros
5. 📐 **Q-Q Plot**: Verifica normalidade dos resíduos
6. 💰 **MAE por Faixa de Preço**: Performance por quartis

## ⚙️ Configurações Personalizáveis

```python
# Diferentes configurações de treinamento
predictor.train_model(
    X_train, X_val, y_train, y_val,
    epochs=200,        # Número de épocas
    batch_size=128     # Tamanho do batch
)

# Diferentes arquiteturas
predictor.build_model(input_dim, architecture='deep')

# Configurações de divisão dos dados
X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_and_scale_data(
    X, y, 
    test_size=0.2,    # 20% para teste
    val_size=0.1      # 10% para validação
)
```

## 📋 Estrutura do Projeto

```
car-price-prediction/
│
├── car_price_predictor.py    # Classe principal
├── requirements.txt          # Dependências
├── README.md                # Este arquivo
├── best_model.keras         # Modelo salvo automaticamente
└── data/
    └── autos.csv           # Dataset (não incluído)
```

## 🎓 Conceitos Técnicos Implementados

- **Batch Normalization**: Acelera convergência e estabiliza treinamento
- **Dropout**: Previne overfitting
- **Early Stopping**: Para treinamento quando não há melhoria
- **Learning Rate Scheduling**: Ajusta taxa de aprendizado automaticamente
- **Huber Loss**: Mais robusta para outliers que MSE
- **Feature Importance**: Análise por permutação
- **Validação Cruzada**: Conjunto separado de validação

## 🔧 Troubleshooting

### Problemas Comuns

**1. Erro de importação do TensorFlow**
```bash
pip install tensorflow --upgrade
```

**2. Problemas com matplotlib no Windows**
```bash
pip install matplotlib --upgrade
```

**3. Dataset não encontrado**
- Verifique o caminho do arquivo
- Certifique-se que o arquivo está em formato CSV
- Verifique a codificação (ISO-8859-1 é padrão)

**4. Memória insuficiente**
- Reduza o `batch_size`
- Use arquitetura 'simple'
- Reduza o número de épocas

## 📊 Requisitos do Sistema

- **Python**: 3.8+
- **RAM**: 4GB mínimo (8GB recomendado)
- **Espaço**: 2GB para dependências
- **GPU**: Opcional (acelera treinamento)

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 👨‍💻 Autor

Desenvolvido com ❤️ para demonstrar aplicações práticas de Deep Learning em problemas reais.

## 📞 Suporte

Se encontrar problemas ou tiver dúvidas:

1. Verifique a seção [Troubleshooting](#-troubleshooting)
2. Abra uma issue no GitHub
3. Consulte a documentação das bibliotecas utilizadas

---

⭐ **Se este projeto foi útil, considere dar uma estrela!** ⭐