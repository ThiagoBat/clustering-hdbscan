"""
Dashboard Streamlit - Polos Gastronômicos de Fortaleza/CE
Visualização interativa dos clusters identificados por HDBSCAN
"""

import pickle
from datetime import datetime
import json

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px

st.set_page_config(
    page_title="Polos Gastronômicos - Fortaleza",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .cluster-card {
        background: #000000;
        padding: 1.8rem;
        border-radius: 15px;
        border-left: 6px solid #FF4B4B;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .info-box {
        background: #000000;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .success-box {
        background: #000000;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_dados():
    """
    Carrega os dados previamente salvos do clustering a partir de um arquivo pickle.
    
    Retorna:
        dict ou None: Dicionário contendo os dados carregados, ou None se o arquivo não existir
        ou ocorrer algum erro ao ler.
    """

    try:
        with open('dados_clustering.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def criar_tabela_clusters(estatisticas):
    """
    Cria um DataFrame contendo informações resumidas dos clusters identificados.
    
    Parâmetros:
        estatisticas (list): Lista de dicionários com estatísticas de cada cluster.
        
    Retorna:
        pd.DataFrame: Tabela com colunas como Polo, Estabelecimentos, Raio,
        Densidade, Avaliação Média, Latitude e Longitude.
    """
    dados_tabela = []
    for stats in estatisticas:
        dados_tabela.append({
            'Polo': stats['nome_polo'],
            'Estabelecimentos': stats['num_estabelecimentos'],
            'Raio (km)': round(stats['raio_km'], 2),
            'Densidade (rest/km²)': round(stats['densidade'], 1),
            'Avaliação Média': round(stats['avg_rating'], 2),
            'Latitude': round(stats['centroide'][0], 4),
            'Longitude': round(stats['centroide'][1], 4)
        })
    return pd.DataFrame(dados_tabela)

def criar_metricas_principais(lugares, estatisticas, clusters):
    """
    Exibe as métricas principais dos estabelecimentos e clusters em colunas no Streamlit.
    
    Parâmetros:
        lugares (list): Lista de dicionários com informações de cada estabelecimento.
        estatisticas (list): Lista de dicionários com estatísticas de cada cluster.
        clusters (np.ndarray): Array com rótulos de cluster para cada estabelecimento.
    
    Exibe:
        Quatro métricas principais:
            1. Total de Estabelecimentos
            2. Polos Gastronômicos (clusters identificados)
            3. Estabelecimentos isolados (outliers)
            4. Avaliação Média dos estabelecimentos
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🍽️ Total de Estabelecimentos",
            value=f"{len(lugares):,}",
            delta="Encontrados na busca",
            help="Total de estabelecimentos identificados"
        )

    with col2:
        st.metric(
            label="🎯 Polos Gastronômicos",
            value=len(estatisticas),
            delta="Identificados por HDBSCAN",
            help="Clusters encontrados"
        )

    with col3:
        num_outliers = np.sum(clusters == -1)
        pct_outliers = (num_outliers / len(clusters)) * 100
        st.metric(
            label="🔸 Isolados",
            value=num_outliers,
            delta=f"{pct_outliers:.1f}%",
            delta_color="inverse",
            help="Estabelecimentos isolados"
        )

    with col4:
        avaliacoes = [p.get("rating", 0) for p in lugares if p.get("rating")]
        avg_rating = sum(avaliacoes) / len(avaliacoes) if avaliacoes else 0
        st.metric(
            label="⭐ Avaliação Média",
            value=f"{avg_rating:.2f}",
            delta="De 5.0 estrelas",
            help="Média geral"
        )

def criar_graficos_estatisticas(estatisticas):
    """
    Gera gráficos interativos de estatísticas dos clusters usando Plotly Express.
    
    Parâmetros:
        estatisticas (list): Lista de dicionários com estatísticas de cada cluster.
        
    Retorna:
        tuple: Três figuras Plotly Express:
            1. Gráfico de barras com o número de estabelecimentos por polo.
            2. Gráfico de barras com a densidade de estabelecimentos por polo.
            3. Gráfico de dispersão relacionando densidade e avaliação média,
            com tamanho proporcional ao número de estabelecimentos.
    """
    df = criar_tabela_clusters(estatisticas)

    fig1 = px.bar(
        df,
        x='Polo',
        y='Estabelecimentos',
        title='📊 Estabelecimentos por Polo',
        color='Estabelecimentos',
        color_continuous_scale='Sunset',
        text='Estabelecimentos'
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(height=400, showlegend=False)

    fig2 = px.bar(
        df,
        x='Polo',
        y='Densidade (rest/km²)',
        title='🔥 Densidade por Polo',
        color='Densidade (rest/km²)',
        color_continuous_scale='Viridis',
        text=[f"{d:.1f}" for d in df['Densidade (rest/km²)']]
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(height=400, showlegend=False)

    fig3 = px.scatter(
        df,
        x='Densidade (rest/km²)',
        y='Avaliação Média',
        size='Estabelecimentos',
        color='Polo',
        title='🔍 Densidade vs Qualidade',
        hover_data=['Raio (km)'],
        size_max=40
    )
    fig3.update_layout(height=500)

    return fig1, fig2, fig3

def criar_mapa_interativo(lugares, estatisticas, clusters, lugares_validos):
    """
    Cria um mapa interativo com os polos gastronômicos e seus estabelecimentos usando Folium.
    
    Parâmetros:
        lugares (list): Lista de dicionários contendo informações de todos os estabelecimentos.
        estatisticas (list): Lista de dicionários com estatísticas de cada polo.
        clusters (array-like): Array com o identificador do cluster de cada estabelecimento.
        lugares_validos (list): Lista de estabelecimentos válidos utilizados no clustering.
        
    Retorna:
        folium.Map: Mapa interativo
    """
    cores = ['red', 'blue', 'green', 'purple', 'orange',
             'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'pink']

    coords = np.array([[p.get("location", {}).get("latitude", 0),
                       p.get("location", {}).get("longitude", 0)] for p in lugares])

    mapa = folium.Map(
        location=[np.mean(coords[:, 0]), np.mean(coords[:, 1])],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    for idx, stats in enumerate(estatisticas):
        centroide = stats['centroide']
        cor = cores[idx % len(cores)]
        nome_polo = stats.get('nome_polo', f'Polo {idx+1}')

        folium.Circle(
            location=[centroide[0], centroide[1]],
            radius=stats['raio_km'] * 1000,
            color=cor,
            fill=True,
            fillColor=cor,
            fillOpacity=0.15,
            opacity=0.6,
            weight=3,
            popup=folium.Popup(f"""
                <div style="width:260px; font-family: Arial;">
                    <h3 style="color: {cor};">🍽️ {nome_polo}</h3>
                    <hr>
                    <p><strong>Estabelecimentos:</strong> {stats['num_estabelecimentos']}</p>
                    <p><strong>Raio:</strong> {stats['raio_km']:.2f} km</p>
                    <p><strong>Densidade:</strong> {stats['densidade']:.1f} rest/km²</p>
                    <p><strong>Avaliação:</strong> {stats['avg_rating']:.2f}/5.0</p>
                </div>
            """, max_width=280)
        ).add_to(mapa)

        folium.Marker(
            location=[centroide[0], centroide[1]],
            icon=folium.Icon(color=cor, icon='cutlery', prefix='fa'),
            popup=f"<b>{nome_polo}</b>",
            tooltip=f"{nome_polo}: {stats['num_estabelecimentos']} estabelecimentos"
        ).add_to(mapa)

    return mapa

def exibir_detalhes_polo(stats):
    """
    Exibe detalhes de um polo gastronômico específico no Streamlit.
    
    Parâmetros:
        stats (dict): Dicionário contendo estatísticas e informações do polo, incluindo:
            - nome_polo (str): Nome do polo.
            - num_estabelecimentos (int): Número de estabelecimentos no polo.
            - avg_rating (float): Avaliação média do polo.
            - raio_km (float): Raio do polo em quilômetros.
            - densidade (float): Densidade de estabelecimentos (restaurantes/km²).
            - centroide (list/tuple): Coordenadas [latitude, longitude]
            do centro do polo.
            - estabelecimentos (list, opcional): Lista de dicionários com
            informações detalhadas de cada estabelecimento.
            
    Funcionalidades:
        - Mostra métricas principais do polo em colunas.
        - Exibe uma tabela dos estabelecimentos do polo, se disponíveis.
        - Permite download da tabela de estabelecimentos em CSV.
    """
    nome_polo = stats.get('nome_polo', 'Polo')

    st.markdown(f"""
    <div class="cluster-card">
        <h2>🍽️ {nome_polo}</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🏪 Estabelecimentos", stats['num_estabelecimentos'])
        st.metric("⭐ Avaliação", f"{stats['avg_rating']:.2f}")

    with col2:
        st.metric("📏 Raio", f"{stats['raio_km']:.2f} km")
        st.metric("📊 Densidade", f"{stats['densidade']:.1f} rest/km²")

    with col3:
        st.metric("📍 Latitude", f"{stats['centroide'][0]:.4f}")
        st.metric("📍 Longitude", f"{stats['centroide'][1]:.4f}")

    if 'estabelecimentos' in stats and len(stats['estabelecimentos']) > 0:
        st.markdown("---")
        st.subheader("📋 Lista de Estabelecimentos")

        est_data = []
        for est in stats['estabelecimentos']:
            est_data.append({
                'Nome': est.get('displayName', {}).get('text', 'Sem nome'),
                'Avaliação': f"{est.get('rating', 0):.1f} ⭐" if est.get('rating') else 'N/A',
                'Endereço': est.get('formattedAddress', 'N/A')
            })

        df_est = pd.DataFrame(est_data)
        st.dataframe(df_est, use_container_width=True, height=350)

        csv = df_est.to_csv(index=False).encode('utf-8')
        st.download_button(
            f"📥 Download - {nome_polo}",
            data=csv,
            file_name=f'{nome_polo.lower().replace(" ", "_")}.csv',
            mime='text/csv'
        )

def main():
    """
    Função principal do dashboard Streamlit para visualização de polos gastronômicos em Fortaleza.
    """
    st.markdown('<div class="main-header">🍽️ Dashboard de Polos Gastronômicos</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fortaleza, Ceará - Análise com HDBSCAN</div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2771/2771436.png", width=150)
        st.markdown("### 🎯 Informações")
        st.markdown("""
        - **Algoritmo:** HDBSCAN  
        - **Fonte:** Google Places API  
        - **Cidade:** Fortaleza, CE  
        - **Tipo:** restaurant, 
        bar, cafe, bakery, meal_takeaway, 
        ice_cream_shop, fast_food_restaurant, 
        pizza_restaurant, sandwich_shop, coffee_shop
        """)
        st.markdown("---")
        st.markdown("### 📊 Status")

    dados = carregar_dados()

    if dados is None:
        st.error("⚠️ **Arquivo de dados não encontrado!**")
        st.markdown("""
        <div class="warning-box">
        <h3>📝 Como usar este dashboard:</h3>
        <ol>
            <li>Execute o script principal: <code>python mapa_calor_hdbscan.py</code></li>
            <li>Os dados serão salvos automaticamente em <code>dados_clustering.pkl</code></li>
            <li>Execute este dashboard: <code>streamlit run dashboard_streamlit.py</code></li>
        </ol>
        <p><strong>Arquivo necessário:</strong> <code>dados_clustering.pkl</code></p>
        </div>
        """, unsafe_allow_html=True)
        return

    lugares = dados['lugares']
    estatisticas = dados['estatisticas']
    clusters = dados['clusters']
    lugares_validos = dados['lugares_validos']

    with st.sidebar:
        st.metric("Polos", len(estatisticas))
        st.metric("Estabelecimentos", len(lugares))
        st.metric("Isolados", np.sum(clusters == -1))

    st.markdown("## 📊 Visão Geral")
    criar_metricas_principais(lugares, estatisticas, clusters)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Estatísticas", 
        "🗺️ Mapa", 
        "📋 Detalhes",
        "📥 Exportar"
    ])

    with tab1:
        st.markdown("### 📈 Análise Estatística")

        fig1, fig2, fig3 = criar_graficos_estatisticas(estatisticas)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)

        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Tabela Resumo")

        df = criar_tabela_clusters(estatisticas)
        st.dataframe(
            df.style.background_gradient(subset=['Densidade (rest/km²)'], cmap='YlOrRd')
                    .background_gradient(subset=['Avaliação Média'], cmap='Greens'),
            use_container_width=True,
            height=400
        )

    with tab2:
        st.markdown("### 🗺️ Mapa Interativo dos Polos")
        st.markdown("""
        <div class="info-box">
        <strong>💡 Dica:</strong> Clique nos marcadores e círculos para ver detalhes de cada polo
        </div>
        """, unsafe_allow_html=True)

        mapa = criar_mapa_interativo(lugares, estatisticas, clusters, lugares_validos)
        folium_static(mapa, width=1400, height=700)

    with tab3:
        st.markdown("### 📋 Detalhes dos Polos Gastronômicos")

        polo_selecionado = st.selectbox(
            "🔍 Selecione um polo para visualizar:",
            [s['nome_polo'] for s in estatisticas],
            help="Escolha um polo para ver informações detalhadas"
        )

        idx = [i for i, s in enumerate(estatisticas) if s['nome_polo'] == polo_selecionado][0]
        st.markdown("---")
        exibir_detalhes_polo(estatisticas[idx])

    with tab4:
        st.markdown("### 📥 Exportar Dados")

        df = criar_tabela_clusters(estatisticas)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Formato CSV")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Tabela Completa (CSV)",
                data=csv,
                file_name=f'polos_fortaleza_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                help="Baixar todos os dados em formato CSV"
            )

        with col2:
            st.markdown("#### 📄 Formato JSON")
            json_data = df.to_dict(orient='records')
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Dados (JSON)",
                data=json_str.encode('utf-8'),
                file_name=f'polos_fortaleza_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json',
                help="Baixar dados estruturados em JSON"
            )

        st.markdown("---")
        st.markdown("""
        <div class="success-box">
        <strong>✅ Formatos Disponíveis:</strong>
        <ul>
            <li><strong>CSV:</strong> Ideal para Excel, Google Sheets e análises</li>
            <li><strong>JSON:</strong> Ideal para APIs e integração com sistemas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1.5rem;">
        <p><strong>Dashboard de Polos Gastronômicos - Fortaleza/CE</strong></p>
        <p>Desenvolvido usando Streamlit | HDBSCAN Clustering</p>
        <p style="font-size: 0.85em; color: #999;">© 2024 - Dados via Google Places API</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
