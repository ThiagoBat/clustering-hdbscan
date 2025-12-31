# 🍽️ Análise de Polos Gastronômicos - RMF/FORTALEZA/CE

Sistema de identificação e análise de clusters utilizando clustering geoespacial (HDBSCAN) e dados do Google Places API. O projeto mapeia estabelecimentos alimentícios em Fortaleza, identifica áreas de concentração e gera análises detalhadas com visualizações interativas.

## 📋 Índice

- [Características](#-características)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Metodologia](#-metodologia)

## ✨ Características

### Análise Geoespacial
- 🗺️ Busca extensiva em grid de coordenadas geográficas
- 🎯 Identificação automática de polos gastronômicos usando HDBSCAN
- 📊 Cálculo de densidade, raio e estatísticas por polo
- 🔍 Detecção de estabelecimentos isolados (outliers)

### Visualizações
- 🌡️ Mapa de calor interativo com Folium
- 📈 Gráficos analíticos (distribuição, densidade, qualidade)
- 🎨 Dashboard HTML completo com métricas e insights
- 📍 Marcadores detalhados para cada polo identificado

### Análise Avançada
- 📊 Métricas de validação de clustering (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- 🧪 Teste automático de múltiplas combinações de parâmetros
- 📉 Análise comparativa de configurações
- 💾 Exportação de dados para visualização externa (Streamlit)

## 🔧 Pré-requisitos

- Python
- Google Maps API Key (Places API habilitada)
- Bibliotecas Python (ver `requirements.txt`)

Dica: Ao cadastrar um cartão no google cloud é disponibilizado um valor para teste gratuito,
esse valor pode ser utilizado para realizar as consultas a API.

## 📦 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/analise-polos-gastronomicos.git
cd analise-polos-gastronomicos
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Arquivo `requirements.txt`
```
requests>=2.28.0
folium>=0.14.0
numpy>=1.23.0
hdbscan>=0.8.29
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

## ⚙️ Configuração

### 1. API Key do Google Maps

Obtenha uma API key no [Google Cloud Console](https://console.cloud.google.com/):
- Crie um projeto
- Ative a "Places API (New)"
- Gere uma API key
- Configure restrições de segurança (opcional, mas recomendado)

### 2. Configure o código

Edite o arquivo principal e insira sua API key:

```python
API_KEY = "SUA_API_KEY_AQUI"
```

### 3. Ajuste a área de busca (opcional)

```python
LATITUDE_MINIMA = -3.9500
LATITUDE_MAXIMA = -3.6100
LONGITUDE_MINIMA = -38.7500
LONGITUDE_MAXIMA = -38.3800
```

### 4. Personalize parâmetros (opcional)

```python
# Tipos de estabelecimentos
TIPOS_ESTABELECIMENTO = [
    "restaurant",
    "bar",
    "cafe",
    "bakery",
    "meal_takeaway",
    "ice_cream_shop",
    "fast_food_restaurant",
    "pizza_restaurant",
    "sandwich_shop",
    "coffee_shop"
]

# Clustering
MIN_CLUSTER_SIZE = 8      # Tamanho mínimo do cluster
MIN_SAMPLES = 20          # Amostras mínimas para densidade

# Grid de busca
PONTOS_GRID = 20          # Resolução do grid (20x20 = 400 pontos)
RAIO_BUSCA = 2500         # Raio de busca por ponto (metros)

# Features
MODO_ANALISE_AVANCADA = True
TESTAR_PARAMETROS = True
GERAR_GRAFICOS = True
GERAR_RELATORIO = True
```

## 🚀 Uso

Execute o script principal:

```bash
python mapa_calor_fortaleza.py
```

## 📊 Resultados

### Arquivos Gerados

```
projeto/
├── mapa_calor_fortaleza_hdbscan.html  # Mapa interativo principal
├── dados_clustering.pkl                # Dados serializados
├── estatisticas_clusters.json          # Estatísticas em JSON
└── analise_avancada/
    ├── 01_distribuicao_clusters.png
    ├── 02_densidade_vs_qualidade.png
    ├── 03_distribuicao_avaliacoes.png
    ├── 04_comparacao_metricas.png
    ├── 05_dispersao_espacial.png
    ├── 06_analise_parametros.png
    └── relatorio_completo.html         # Dashboard HTML completo
```

### Visualizações

#### Mapa Interativo
- Mapa de calor mostrando densidade de estabelecimentos
- Círculos coloridos delimitando cada polo
- Marcadores nos centros dos polos com estatísticas
- Pontos cinzas para estabelecimentos isolados

#### Gráficos
- Distribuição de estabelecimentos por polo
- Densidade vs qualidade (scatter plot)
- Histograma de avaliações
- Comparação de métricas (barras horizontais)
- Dispersão espacial com clusters coloridos
- Análise de parâmetros de clustering

#### Relatório HTML
- Métricas principais em cards destacados
- Tabela detalhada de todos os polos
- Visualizações integradas
- Recomendações e insights automáticos
- Metodologia e parâmetros utilizados

## 🔬 Metodologia

### HDBSCAN Clustering

O projeto utiliza **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise):

**Vantagens:**
- Identifica automaticamente o número de clusters
- Robusto a ruído (detecta outliers)
- Não assume formas geométricas específicas
- Ideal para dados geoespaciais

**Métrica:** Haversine
- Calcula distâncias reais sobre a superfície terrestre
- Resultados precisos em quilômetros

### Métricas de Validação

1. **Silhouette Score** (0 a 1)
   - Mede a separação entre clusters
   - Valores próximos a 1 = clusters bem definidos

2. **Davies-Bouldin Index** (menor é melhor)
   - Avalia compactação e separação
   - Valores baixos = clusters distintos

3. **Calinski-Harabasz Score** (maior é melhor)
   - Razão dispersão inter/intra cluster
   - Valores altos = clusters bem separados

## 👤 Autor

Thiago Ramos Batista
- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- LinkedIn: [seu-perfil](https://linkedin.com/in/seu-perfil)