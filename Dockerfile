# Imagem base Python oficial otimizada para produção
FROM python:3.11-slim

WORKDIR /app

# Variáveis de ambiente para Python e Streamlit
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000 \
    STREAMLIT_SERVER_PORT=10000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Instalação de utilitários mínimos do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependências primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código-fonte, modelo serializado e imagens da aplicação
COPY . .

EXPOSE 10000

# Monitoramento de saúde nativo do Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-10000}/_stcore/health || exit 1

# Comando de inicialização com porta dinâmica gerenciada pelo Render
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-10000} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
