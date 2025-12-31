"""
Este módulo extrai e analisa dados obtidos através da API do google maps 
realiza a clusterização através do HDBSCANN, gera gráficos,
relatório HTML e Streamlit
"""
from datetime import datetime
import os
import time
import pickle
import json
import webbrowser
from math import radians, cos, sin, asin, sqrt

import matplotlib.pyplot as plt
import seaborn as sns
import requests
import folium
from folium.plugins import HeatMap
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib
matplotlib.use('Agg')

API_KEY = ""

LATITUDE_MINIMA = -3.9500
LATITUDE_MAXIMA = -3.6100
LONGITUDE_MINIMA = -38.7500
LONGITUDE_MAXIMA = -38.3800

AREA_CENTER = {
    "latitude": (LATITUDE_MINIMA + LATITUDE_MAXIMA) / 2,
    "longitude": (LONGITUDE_MINIMA + LONGITUDE_MAXIMA) / 2
}

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

RAIO_BUSCA = 2500
PONTOS_GRID = 20

MIN_CLUSTER_SIZE = 8
MIN_SAMPLES = 20

MODO_ANALISE_AVANCADA = True
MOSTRAR_CLUSTERS = True
MOSTRAR_OUTLIERS = True

TESTAR_PARAMETROS = True
GERAR_GRAFICOS = True
GERAR_RELATORIO = True

MAX_TENTATIVAS = 5
BACKOFF_INICIAL = 1

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância geodésica entre dois pontos da superfície da Terra
    utilizando a fórmula de Haversine.

    Args:
    lat1 (float): Latitude do primeiro ponto em graus decimais.
    lon1 (float): Longitude do primeiro ponto em graus decimais.
    lat2 (float): Latitude do segundo ponto em graus decimais.
    lon2 (float): Longitude do segundo ponto em graus decimais.

    Returns:
    float: Distância entre os dois pontos em quilômetros.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def buscar_estabelecimentos(session, api_key, lat, lng, tipos, raio,
                            max_results=20):
    """
    Busca estabelecimentos próximos a uma coordenada geográfica utilizando
    a API Google Places (Nearby Search).
    
    Args:
        session (requests.Session): Sessão HTTP reutilizável para realizar as requisições.
        api_key (str): Chave de autenticação da API Google Places.
        lat (float): Latitude do ponto central da busca em graus decimais.
        lng (float): Longitude do ponto central da busca em graus decimais.
        tipos (list[str]): Lista de tipos de estabelecimentos a serem incluídos na busca.
        raio (float): Raio da busca em metros a partir do ponto central.
        max_results (int, optional): Número máximo de estabelecimentos retornados.
        
    Returns:
        list: Lista de estabelecimentos retornados pela API, contendo informações
        como identificação, nome, localização, endereço, tipos, avaliação e status
        do negócio.
        """
    url = "https://places.googleapis.com/v1/places:searchNearby"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.location,"
            "places.formattedAddress,places.types,"
            "places.rating,places.businessStatus"
            )
    }

    body = {
        "includedTypes": tipos,
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": raio
            }
        }
    }

    tentativas = 0
    backoff = BACKOFF_INICIAL

    while tentativas < MAX_TENTATIVAS:
        try:
            response = session.post(url, headers=headers, json=body, timeout=15)

            if response.status_code == 429:
                print(f"Rate limit, aguardando {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                tentativas += 1
                continue

            if response.status_code >= 500:
                print(f"Erro do servidor ({response.status_code}), aguardando {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                tentativas += 1
                continue

            response.raise_for_status()
            data = response.json()
            time.sleep(0.2)
            return data.get("places", [])

        except requests.exceptions.Timeout:
            tentativas += 1
            print(f"Timeout ({tentativas}/{MAX_TENTATIVAS}), aguardando {backoff}s...")
            time.sleep(backoff)
            backoff *= 2

        except requests.exceptions.RequestException as e:
            tentativas += 1
            print(f"Erro ({tentativas}/{MAX_TENTATIVAS}): {e}")
            if tentativas < MAX_TENTATIVAS:
                time.sleep(backoff)
                backoff *= 2

    print(f"Falha após {MAX_TENTATIVAS} tentativas")
    return []

def buscar_area_extensa(api_key, lat_min, lat_max, lng_min, lng_max, tipos,
                        raio, pontos_grid):
    """
    Busca estabelecimentos em uma área geográfica extensa a partir de um grid
    de pontos, realizando múltiplas consultas à API Google Places e consolidando
    os resultados em uma lista única sem duplicidades.
    
    Args:
        api_key (str): Chave de autenticação da API Google Places.
        lat_min (float): Latitude mínima da área de busca em graus decimais.
        lat_max (float): Latitude máxima da área de busca em graus decimais.
        lng_min (float): Longitude mínima da área de busca em graus decimais.
        lng_max (float): Longitude máxima da área de busca em graus decimais.
        tipos (list[str]): Lista de tipos de estabelecimentos a serem buscados.
        raio (float): Raio de busca em metros para cada ponto do grid.
        pontos_grid (int): Número de divisões do grid em cada eixo
        (latitude e longitude).
        
    Returns:
        list: Lista de estabelecimentos únicos encontrados na área definida,
        considerando deduplicação por place_id ou coordenadas geográficas.
        """

    session = requests.Session()
    todos_lugares = []
    lugares_por_id = {}
    lugares_sem_id = set()

    filtrados_por_status = 0

    lats = []
    lngs = []

    for i in range(pontos_grid):
        lat = lat_min + (lat_max - lat_min) * i / (pontos_grid - 1)
        lats.append(lat)

    for i in range(pontos_grid):
        lng = lng_min + (lng_max - lng_min) * i / (pontos_grid - 1)
        lngs.append(lng)

    total_pontos = pontos_grid * pontos_grid

    print("ÁREA DE BUSCA DEFINIDA")
    print(f"Latitude:  {lat_min:.4f} a {lat_max:.4f}")
    print(f"Longitude: {lng_min:.4f} a {lng_max:.4f}")
    print(f"Grid: {pontos_grid}x{pontos_grid} = {total_pontos} pontos de busca")
    print(f"Raio por ponto: {raio}m")
    print(f"Tipos: {', '.join(tipos)}")

    if total_pontos > 50:
        tempo_estimado = total_pontos * 1.5 / 60
        print(f"Tempo estimado: ~{tempo_estimado:.1f} minutos")

    print()

    ponto_atual = 0
    inicio = time.time()

    for lat in lats:
        for lng in lngs:
            ponto_atual += 1

            if ponto_atual > 1:
                tempo_decorrido = time.time() - inicio
                tempo_por_ponto = tempo_decorrido / (ponto_atual - 1)
                tempo_restante = tempo_por_ponto * (total_pontos - ponto_atual)
                print(f"[{ponto_atual}/{total_pontos}] ({lat:.4f}, {lng:.4f}) |"
                      f"ETA: {tempo_restante/60:.1f}min...", end=" ")
            else:
                print(f"[{ponto_atual}/{total_pontos}] ({lat:.4f}, {lng:.4f})...", end=" ")

            novos_neste_ponto = 0

            for tipo in tipos:
                print(f"-> Buscando tipo: {tipo}")

                lugares = buscar_estabelecimentos(
                    session,
                    api_key,
                    lat,
                    lng,
                    [tipo],
                    raio,
                    20
                )

                print(f"  {tipo}: {len(lugares)} retornados")

                for lugar in lugares:
                    business_status = lugar.get("businessStatus", "OPERATIONAL")
                    if business_status != "OPERATIONAL":
                        filtrados_por_status += 1
                        continue

                    place_id = lugar.get("id")

                    if place_id and place_id in lugares_por_id:
                        tipo_existente = lugares_por_id[place_id].get("tipo_busca", "")
                        if tipo not in tipo_existente:
                            lugares_por_id[place_id]["tipo_busca"] += f", {tipo}"
                        continue

                    lugar["tipo_busca"] = tipo

                    if place_id:
                        lugares_por_id[place_id] = lugar
                        todos_lugares.append(lugar)
                        novos_neste_ponto += 1
                    else:
                        loc = lugar.get("location", {})
                        lat_lugar = loc.get("latitude")
                        lng_lugar = loc.get("longitude")

                        if lat_lugar and lng_lugar:
                            identificador = f"{lat_lugar:.6f},{lng_lugar:.6f}"
                            if identificador not in lugares_sem_id:
                                lugares_sem_id.add(identificador)
                                todos_lugares.append(lugar)
                                novos_neste_ponto += 1

            print(f"  Novos neste ponto: {novos_neste_ponto} |"
                  f"Total acumulado: {len(todos_lugares)}\n")

    tempo_total = time.time() - inicio
    session.close()

    print()
    print("BUSCA CONCLUÍDA!")
    print(f"   Tempo total: {tempo_total/60:.1f} minutos")
    print(f"   Requisições: {total_pontos * len(tipos)}")
    print(f"   Estabelecimentos únicos: {len(todos_lugares)}")
    print(f"   (Deduplicados por place_id: {len(lugares_por_id)},"
          f"por coordenadas: {len(lugares_sem_id)})")
    print(f"   Filtrados por status não-operacional: {filtrados_por_status}")
    print()

    return todos_lugares

def identificar_polos_gastronomicos_hdbscan(lugares,
                                            min_cluster_size,
                                            min_samples,
                                            verbose=True):
    """
    Identifica polos gastronômicos a partir da distribuição espacial de
    estabelecimentos utilizando o algoritmo de clusterização HDBSCAN
    com métrica de Haversine.
    
    Args:
        lugares (list): Lista de estabelecimentos
        contendo informações de localização.
        min_cluster_size (int): Número mínimo de estabelecimentos
        para formar um cluster.
        min_samples (int): Número mínimo de amostras para um ponto
        ser considerado núcleo.
        verbose (bool): Indica se mensagens informativas devem ser
        exibidas durante a execução.
        
    Returns:
        tuple: Tupla contendo três elementos:
            - clusters (numpy.ndarray): Rótulos de cluster atribuídos a 
            cada estabelecimento.
            - estatisticas (list): Lista de dicionários com métricas dos 
            polos identificados, incluindo centroide, raio,
            densidade, avaliação média e estabelecimentos.
            - lugares_validos (list): Lista de estabelecimentos
            utilizados efetivamente no processo de clusterização.
    """
    if len(lugares) < min_cluster_size:
        if verbose:
            print(f"Poucos estabelecimentos ({len(lugares)}) para clustering")
        return None, None, []

    coords = []
    lugares_validos = []

    for lugar in lugares:
        loc = lugar.get("location", {})
        lat = loc.get("latitude")
        lng = loc.get("longitude")
        if lat and lng:
            coords.append([lat, lng])
            lugares_validos.append(lugar)

    coords = np.array(coords)

    coords_rad = np.radians(coords)

    if verbose:
        print("Executando HDBSCAN...")
        print(f"   Tamanho mínimo do cluster: {min_cluster_size}")
        print(f"   Amostras mínimas: {min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='haversine',
        cluster_selection_method='eom'
    )

    clusters = clusterer.fit_predict(coords_rad)

    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    num_outliers = np.sum(clusters == -1)

    if verbose:
        print("Clustering concluído!")
        print(f"   Polos identificados: {num_clusters}")
        print(f"   Estabelecimentos isolados: {num_outliers}")

    if num_clusters == 0:
        if verbose:
            print("Nenhum polo foi identificado. Tente ajustar os parâmetros.")
        return clusters, None, []

    estatisticas = []

    for cluster_id in range(num_clusters):
        indices = np.where(clusters == cluster_id)[0]

        estabelecimentos_cluster = [lugares_validos[j] for j in indices]

        avaliacoes = [p.get("rating", 0) for p in estabelecimentos_cluster if p.get("rating")]
        avg_rating = sum(avaliacoes) / len(avaliacoes) if avaliacoes else 0

        coords_cluster = coords[indices]
        centroide = np.mean(coords_cluster, axis=0)

        distancias = []
        for coord in coords_cluster:
            dist = haversine(centroide[0], centroide[1], coord[0], coord[1])
            distancias.append(dist)

        raio_km = max(distancias) if distancias else 0
        area_km2 = np.pi * (raio_km ** 2) if raio_km > 0 else 0.01
        densidade = len(indices) / area_km2 if area_km2 > 0 else 0

        nome_polo = f"Polo {cluster_id + 1}"

        estatisticas.append({
            'cluster_id': cluster_id,
            'num_estabelecimentos': len(indices),
            'centroide': centroide,
            'avg_rating': avg_rating,
            'raio_km': raio_km,
            'densidade': densidade,
            'estabelecimentos': estabelecimentos_cluster,
            'nome_polo': nome_polo
        })

    estatisticas.sort(key=lambda x: x['densidade'], reverse=True)

    for idx, stats in enumerate(estatisticas, 1):
        stats['nome_polo'] = f"Polo {idx}"

    return clusters, estatisticas, lugares_validos

def calcular_METRICAS_qualidade(coords_km, clusters):
    """
    Calcula métricas de qualidade para avaliação de resultados de clusterização,
    desconsiderando pontos classificados como ruído.
    
    Args:
        coords_km (numpy.ndarray): Array de coordenadas espaciais em quilômetros
        utilizadas no processo de clusterização.
        clusters (numpy.ndarray): Vetor de rótulos de cluster atribuídos a
        cada ponto, onde o valor -1 indica ruído.
        
    Returns:
        dict: Dicionário contendo métricas de avaliação da clusterização,
        incluindo silhouette, davies_bouldin e calinski_harabasz.
        Retorna None caso não seja possível calcular as métricas.
    """
    mask = clusters != -1

    if np.sum(mask) < 2:
        return None

    coords_filtered = coords_km[mask]
    labels_filtered = clusters[mask]

    unique_labels = np.unique(labels_filtered)
    if len(unique_labels) < 2:
        return None

    METRICAS = {}

    try:
        METRICAS['silhouette'] = silhouette_score(coords_filtered,
                                                  labels_filtered)
    except:
        METRICAS['silhouette'] = None

    try:
        METRICAS['davies_bouldin'] = davies_bouldin_score(coords_filtered,
                                                          labels_filtered)
    except:
        METRICAS['davies_bouldin'] = None

    try:
        METRICAS['calinski_harabasz'] = calinski_harabasz_score(
            coords_filtered,
            labels_filtered)
    except:
        METRICAS['calinski_harabasz'] = None

    return METRICAS

def testar_parametros_hdbscan(lugares):
    """
    Testa diferentes combinações de parâmetros do algoritmo HDBSCAN para
    identificação de polos gastronômicos, avaliando a qualidade da
    clusterização por meio de métricas estatísticas.
    
    Args:
        lugares (list): Lista de estabelecimentos contendo informações
        de localização (latitude e longitude).
        
    Returns:
        list: Lista de dicionários com os resultados de cada combinação testada,
        incluindo parâmetros utilizados, número de clusters, quantidade e
        percentual de outliers e métricas de qualidade da clusterização.
    """
    print()
    print("TESTE DE PARÂMETROS HDBSCAN")
    print()

    coords = []
    lugares_validos = []

    for lugar in lugares:
        loc = lugar.get("location", {})
        lat = loc.get("latitude")
        lng = loc.get("longitude")
        if lat and lng:
            coords.append([lat, lng])
            lugares_validos.append(lugar)

    coords = np.array(coords)

    lat_ref = coords[0, 0]
    coords_km = np.zeros_like(coords)

    for i, (lat, lng) in enumerate(coords):
        coords_km[i, 0] = haversine(lat_ref, coords[0, 1], lat, coords[0, 1])
        coords_km[i, 1] = haversine(lat_ref, coords[0, 1], lat_ref, lng)

        if lat < lat_ref:
            coords_km[i, 0] *= -1
        if lng < coords[0, 1]:
            coords_km[i, 1] *= -1

    min_cluster_sizes = [5, 8, 10, 12]
    min_samples_list = [15, 18, 20, 22, 25]

    resultados = []
    total_testes = len(min_cluster_sizes) * len(min_samples_list)
    teste_atual = 0

    print(f"Testando {total_testes} combinações de parâmetros...")
    print()

    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            teste_atual += 1
            print(f"[{teste_atual}/{total_testes}] min_cluster_size={mcs},"
                  f"min_samples={ms}...", end=" ")

            clusters, estatisticas, _ = identificar_polos_gastronomicos_hdbscan(
                lugares, mcs, ms, verbose=False
            )

            if estatisticas is None or len(estatisticas) == 0:
                print("Sem clusters")
                continue

            METRICAS = calcular_METRICAS_qualidade(coords_km, clusters)

            num_clusters = len(estatisticas)
            num_outliers = np.sum(clusters == -1)
            pct_outliers = (num_outliers / len(clusters)) * 100

            resultado = {
                'min_cluster_size': mcs,
                'min_samples': ms,
                'num_clusters': num_clusters,
                'num_outliers': num_outliers,
                'pct_outliers': pct_outliers,
                'silhouette': METRICAS['silhouette'] if METRICAS else None,
                'davies_bouldin': METRICAS['davies_bouldin'] if METRICAS else None,
                'calinski_harabasz': METRICAS['calinski_harabasz'] if METRICAS else None
            }

            resultados.append(resultado)
            print(f"- {num_clusters} clusters, {num_outliers} outliers")

    print()
    print("-> MELHORES CONFIGURAÇÕES")
    print()

    resultados_validos = [r for r in resultados if r['silhouette'] is not None]

    if resultados_validos:
        resultados_validos.sort(key=lambda x: x['silhouette'], reverse=True)

        print("TOP 5 - Por Silhouette Score:")
        print()
        for i, r in enumerate(resultados_validos[:5], 1):
            print(f"{i}. min_cluster_size={r['min_cluster_size']},"
                  f"min_samples={r['min_samples']}")
            print(f"   Clusters: {r['num_clusters']} |"
                  f"Outliers: {r['pct_outliers']:.1f}%")
            print(f"   Silhouette: {r['silhouette']:.3f} |"
                  f"Davies-Bouldin: {r['davies_bouldin']:.3f}")
            print()

    return resultados

def gerar_graficos_analise(lugares, clusters, estatisticas, METRICAS,
                           RESULTADOS_PARAMETROS=None):
    """
    Gera gráficos analíticos para avaliação dos polos gastronômicos identificados,
    incluindo distribuição de estabelecimentos, densidade, qualidade, métricas
    espaciais, dispersão geográfica e análise de parâmetros do HDBSCAN.
    
    Args:
        lugares (list): Lista de estabelecimentos com informações de
        localização e avaliação.
        clusters (numpy.ndarray): Vetor de rótulos de cluster atribuídos
        a cada estabelecimento.
        estatisticas (list): Lista de dicionários contendo métricas dos
        polos identificados, como densidade, raio, centroide e avaliação média.
        METRICAS (dict): Dicionário com métricas globais
        de qualidade da clusterização.
        RESULTADOS_PARAMETROS (list, optional): Resultados dos testes de
        parâmetros do HDBSCAN para análise comparativa.
        
    Returns:
        None
    """

    print()
    print("GERANDO GRÁFICOS DE ANÁLISE")
    print()

    os.makedirs('analise_avancada', exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 10)

    print("1/6 Distribuição por cluster...")
    fig, ax = plt.subplots(figsize=(12, 8))

    num_estabelecimentos = [s['num_estabelecimentos'] for s in estatisticas]
    nomes_clusters = [s.get('nome_polo', f"Polo {i+1}")
                      for i, s in enumerate(estatisticas)]

    cmap = plt.get_cmap('Set3')
    colors = cmap(np.linspace(0, 1, len(estatisticas)))
    bars = ax.bar(range(len(estatisticas)), num_estabelecimentos,
                  color=colors, width=0.7)

    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_ylabel('Número de Estabelecimentos', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{nomes_clusters[i]}\n{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.margins(x=0.02)  # Reduz margem horizontal

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    y_max = max(num_estabelecimentos)
    ax.set_ylim(0, y_max * 1.25)

    plt.tight_layout()
    plt.savefig('analise_avancada/01_distribuicao_clusters.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    print("2/6 Densidade vs Qualidade...")
    fig, ax = plt.subplots(figsize=(10, 6))

    densidades = [s['densidade'] for s in estatisticas]
    avaliacoes = [s['avg_rating'] for s in estatisticas]
    tamanhos = [s['num_estabelecimentos'] * 10 for s in estatisticas]

    scatter = ax.scatter(densidades, avaliacoes, s=tamanhos, alpha=0.6,
                        c=range(len(estatisticas)), cmap='viridis',
                        edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Densidade (restaurantes/km²)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Avaliação Média', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label='Índice do Polo')
    plt.tight_layout()
    plt.savefig('analise_avancada/02_densidade_vs_qualidade.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    print("3/6 Distribuição de avaliações...")
    fig, ax = plt.subplots(figsize=(10, 6))

    todas_avaliacoes = [p.get("rating", 0) for p in lugares if p.get("rating")]

    ax.hist(todas_avaliacoes, bins=20, color='skyblue',
            edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(todas_avaliacoes), color='red', linestyle='--',
               linewidth=2, label=f'Média: {np.mean(todas_avaliacoes):.2f}')
    ax.axvline(np.median(todas_avaliacoes), color='green', linestyle='--',
               linewidth=2, label=f'Mediana: {np.median(todas_avaliacoes):.2f}')

    ax.set_xlabel('Avaliação', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
    ax.set_title('Distribuição de Avaliações dos Estabelecimentos',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analise_avancada/03_distribuicao_avaliacoes.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("4/6 Comparação de métricas...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    indices_polos = range(len(estatisticas))

    axes[0, 0].barh(indices_polos, densidades, color='coral')
    axes[0, 0].set_xlabel('Densidade (rest/km²)', fontweight='bold')
    axes[0, 0].set_title('Densidade por Polo', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    axes[0, 1].barh(indices_polos, avaliacoes, color='lightgreen')
    axes[0, 1].set_xlabel('Avaliação Média', fontweight='bold')
    axes[0, 1].set_title('Qualidade por Polo', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    raios = [s['raio_km'] for s in estatisticas]
    axes[1, 0].barh(indices_polos, raios, color='lightblue')
    axes[1, 0].set_xlabel('Raio (km)', fontweight='bold')
    axes[1, 0].set_title('Tamanho dos Polos', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    axes[1, 1].barh(indices_polos, num_estabelecimentos, color='plum')
    axes[1, 1].set_xlabel('Quantidade', fontweight='bold')
    axes[1, 1].set_title('Estabelecimentos por Polo', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('analise_avancada/04_comparacao_METRICAS.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("5/6 Mapa de dispersão espacial...")
    fig, ax = plt.subplots(figsize=(12, 10))

    coords = np.array([[p.get("location", {}).get("latitude", 0),
                       p.get("location", {}).get("longitude", 0)]
                       for p in lugares])

    unique_clusters = np.unique(clusters)
    cmap = plt.get_cmap('tab10')
    colors_map = cmap(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1:
            mask = clusters == cluster_id
            ax.scatter(coords[mask, 1], coords[mask, 0],
                      c='lightgray', s=30, alpha=0.5,
                      label='Isolados', marker='x')
        else:
            mask = clusters == cluster_id
            ax.scatter(coords[mask, 1], coords[mask, 0],
                      c=[colors_map[i]], s=50, alpha=0.7,
                      label=f'Polo {cluster_id+1}', edgecolors='black',
                      linewidth=0.5)

    for stats in estatisticas:
        ax.scatter(stats['centroide'][1], stats['centroide'][0],
                  c='red', s=300, marker='*', edgecolors='black',
                  linewidth=2, zorder=5)

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('Distribuição Espacial dos Clusters', fontsize=14,
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analise_avancada/05_dispersao_espacial.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    if RESULTADOS_PARAMETROS:
        print("6/6 Análise de parâmetros...")

        resultados_validos = [r for r in RESULTADOS_PARAMETROS
                              if r['silhouette'] is not None]

        if len(resultados_validos) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            num_clusters_list = [r['num_clusters'] for r in resultados_validos]
            silhouettes = [r['silhouette'] for r in resultados_validos]

            axes[0, 0].scatter(num_clusters_list, silhouettes, alpha=0.6,
                               s=100, c='blue')
            axes[0, 0].set_xlabel('Número de Clusters', fontweight='bold')
            axes[0, 0].set_ylabel('Silhouette Score', fontweight='bold')
            axes[0, 0].set_title('Qualidade vs Número de Clusters',
                                 fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)

            davies = [r['davies_bouldin'] for r in resultados_validos]
            axes[0, 1].scatter(num_clusters_list, davies, alpha=0.6, s=100,
                               c='red')
            axes[0, 1].set_xlabel('Número de Clusters', fontweight='bold')
            axes[0, 1].set_ylabel('Davies-Bouldin Index', fontweight='bold')
            axes[0, 1].set_title('Compactação vs Número de Clusters',
                                 fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)

            min_cluster_sizes_unique = sorted(list(set([r['min_cluster_size']
                                                        for r in
                                                        resultados_validos])))
            outliers_por_mcs = {}
            for mcs in min_cluster_sizes_unique:
                outliers = [r['pct_outliers'] for r in resultados_validos
                            if r['min_cluster_size'] == mcs]
                outliers_por_mcs[mcs] = np.mean(outliers)

            axes[1, 0].bar(outliers_por_mcs.keys(), outliers_por_mcs.values(),
                           color='orange')
            axes[1, 0].set_xlabel('Min Cluster Size', fontweight='bold')
            axes[1, 0].set_ylabel('% Médio de Outliers', fontweight='bold')
            axes[1, 0].set_title('Impacto do Tamanho Mínimo',
                                 fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            top5 = sorted(resultados_validos, key=lambda x: x['silhouette'],
                          reverse=True)[:5]
            labels = [f"mcs={r['min_cluster_size']}\nms={r['min_samples']}"
                      for r in top5]
            scores = [r['silhouette'] for r in top5]

            axes[1, 1].barh(range(len(labels)), scores, color='green')
            axes[1, 1].set_yticks(range(len(labels)))
            axes[1, 1].set_yticklabels(labels, fontsize=8)
            axes[1, 1].set_xlabel('Silhouette Score', fontweight='bold')
            axes[1, 1].set_title('Top 5 Configurações', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig('analise_avancada/06_analise_parametros.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    print()
    print("Gráficos salvos em: analise_avancada/")
    print()

def gerar_relatorio_html(lugares, clusters, estatisticas, METRICAS,
                         RESULTADOS_PARAMETROS=None):
    """
    Gera um relatório analítico em formato HTML contendo a síntese dos resultados
    da identificação de polos gastronômicos, incluindo métricas de clusterização,
    tabelas descritivas, gráficos e recomendações.
    
    Args:
        lugares (list): Lista de estabelecimentos com informações de
        localização, avaliação e atributos gerais.
        clusters (numpy.ndarray): Vetor de rótulos de cluster atribuídos a
        cada estabelecimento, onde o valor -1 indica estabelecimentos isolados.
        estatisticas (list): Lista de dicionários contendo métricas e
        características dos polos identificados, como centroide, densidade,
        raio, avaliação média e quantidade de estabelecimentos.
        METRICAS (dict): Dicionário com métricas globais de qualidade da
        clusterização (silhouette, davies_bouldin e calinski_harabasz).
        RESULTADOS_PARAMETROS (list, optional): Resultados dos testes de
        parâmetros do HDBSCAN para inclusão de análises
        comparativas no relatório.
    
    Returns:
        None
    """

    print()
    print("GERANDO RELATÓRIO HTML")
    print()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Análise - Polos Gastronômicos Fortaleza</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
               line-height: 1.6; color: #333; background:
                   linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                   padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white;
                     border-radius: 15px; box-shadow:
                         0 10px 40px rgba(0,0,0,0.3); overflow: hidden; }}
        .header {{ background:
                  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                  color: white; padding: 40px;
                                  text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ color: #667eea; border-bottom: 3px solid #667eea;
                      padding-bottom: 10px; margin-bottom: 20px;
                      font-size: 1.8em; }}
        .metrics-grid {{ display: grid; grid-template-columns:
                        repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;
                        margin-bottom: 30px; }}
        .metric-card {{ background:
                       linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                       color: white; padding: 25px;
                                       border-radius: 10px;
                                       box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                                       text-align: center;
                                       transition: transform 0.3s ease; }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; margin: 10px 0; }}
        .metric-label {{ font-size: 1em; opacity: 0.9; }}
        .cluster-table {{ width: 100%;
                         border-collapse: collapse;
                         margin: 20px 0;
                         box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .cluster-table th {{ background:
                            linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                            color: white; padding: 15px;
                                            text-align: left;
                                            font-weight: bold; }}
        .cluster-table td {{ padding: 12px 15px;
                            border-bottom: 1px solid #ddd; }}
        .cluster-table tr:hover {{ background-color: #f5f5f5; }}
        .cluster-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .quality-badge {{ display: inline-block; padding: 5px 15px;
                         border-radius: 20px;
                         font-weight: bold; font-size: 0.9em; }}
        .quality-high {{ background-color: #4CAF50; color: white; }}
        .quality-medium {{ background-color: #FF9800; color: white; }}
        .quality-low {{ background-color: #f44336; color: white; }}
        .chart-container {{ margin: 30px 0; text-align: center; }}
        .chart-container img {{ max-width: 100%; height: auto;
                               border-radius: 10px;
                               box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .info-box {{ background: #e3f2fd;
                    border-left: 4px solid #2196F3;
                    padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .success-box {{ background: #e8f5e9;
                       border-left: 4px solid #4CAF50;
                       padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .warning-box {{ background: #fff3e0; border-left: 4px solid #FF9800;
                       padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .footer {{ background: #f5f5f5; padding: 30px;
                  text-align: center; color: #666;
                  border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🍽️ Relatório de Análise de Polos Gastronômicos</h1>
            <p>Fortaleza, Ceará - Brasil</p>
            <p style="font-size: 0.9em;
            margin-top: 10px;">Gerado em: {timestamp}</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📊 Métricas Principais</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total de Estabelecimentos</div>
                        <div class="metric-value">{len(lugares)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Polos Identificados</div>
                        <div class="metric-value">{len(estatisticas)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Estabelecimentos Isolados</div>
                        <div class="metric-value">{np.sum(clusters == -1)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avaliação Média</div>
                        <div class="metric-value">{np.mean([p.get("rating", 0)
                                                            for p in lugares if p.get("rating")]):.2f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Qualidade do Clustering</h2>
"""

    if METRICAS:
        silhouette_class = "quality-high" if METRICAS.get('silhouette', 0) > 0.5 else "quality-medium" if METRICAS.get('silhouette', 0) > 0.3 else "quality-low"

        html += f"""
                <div class="success-box">
                    <h3>Métricas de Validação</h3>
                    <p><strong>Silhouette Score:</strong> <span class="quality-badge {silhouette_class}">{METRICAS.get('silhouette', 0):.3f}</span></p>
                    <p style="margin-top: 10px;"><em>Indica quão bem separados estão os clusters. Valores próximos de 1 são ideais.</em></p>
                    <p style="margin-top: 20px;"><strong>Davies-Bouldin Index:</strong> <strong>{METRICAS.get('davies_bouldin', 0):.3f}</strong></p>
                    <p style="margin-top: 10px;"><em>Mede a compactação dos clusters. Valores menores são melhores.</em></p>
                    <p style="margin-top: 20px;"><strong>Calinski-Harabasz Score:</strong> <strong>{METRICAS.get('calinski_harabasz', 0):.1f}</strong></p>
                    <p style="margin-top: 10px;"><em>Razão entre dispersão inter e intra-cluster. Valores maiores são melhores.</em></p>
                </div>
"""

    html += """
            </div>
            
            <div class="section">
                <h2>🎯 Detalhes dos Polos Gastronômicos</h2>
                <table class="cluster-table">
                    <thead>
                        <tr>
                            <th>Polo</th>
                            <th>Estabelecimentos</th>
                            <th>Densidade (rest/km²)</th>
                            <th>Raio (km)</th>
                            <th>Avaliação Média</th>
                            <th>Qualidade</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for stats in estatisticas:
        nome_polo = stats.get('nome_polo', 'Polo')
        rating = stats['avg_rating']
        quality_class = "quality-high" if rating >= 4.0 else "quality-medium" if rating >= 3.5 else "quality-low"

        html += f"""
                        <tr>
                            <td><strong>{nome_polo}</strong></td>
                            <td>{stats['num_estabelecimentos']}</td>
                            <td>{stats['densidade']:.1f}</td>
                            <td>{stats['raio_km']:.2f}</td>
                            <td>{stats['avg_rating']:.2f} ⭐</td>
                            <td><span class="quality-badge {quality_class}">{rating:.2f}</span></td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📈 Visualizações</h2>
                <div class="chart-container">
                    <h3>Distribuição de Estabelecimentos por Polo</h3>
                    <img src="01_distribuicao_clusters.png" alt="Distribuição">
                </div>
                <div class="chart-container">
                    <h3>Densidade vs Qualidade</h3>
                    <img src="02_densidade_vs_qualidade.png" alt="Densidade vs Qualidade">
                </div>
                <div class="chart-container">
                    <h3>Distribuição de Avaliações</h3>
                    <img src="03_distribuicao_avaliacoes.png" alt="Avaliações">
                </div>
                <div class="chart-container">
                    <h3>Comparação de Métricas</h3>
                    <img src="04_comparacao_METRICAS.png" alt="Comparação">
                </div>
                <div class="chart-container">
                    <h3>Distribuição Espacial</h3>
                    <img src="05_dispersao_espacial.png" alt="Dispersão">
                </div>
"""

    if RESULTADOS_PARAMETROS:
        html += """
                <div class="chart-container">
                    <h3>Análise de Parâmetros</h3>
                    <img src="06_analise_parametros.png" alt="Parâmetros">
                </div>
"""

    html += """
            </div>
            
            <div class="section">
                <h2>💡 Recomendações e Insights</h2>
                <div class="info-box">
                    <h3>🎯 Polos Principais</h3>
"""

    top3_densidade = sorted(estatisticas, key=lambda x: x['densidade'],
                            reverse=True)[:3]
    for i, stats in enumerate(top3_densidade, 1):
        html += f"""
                    <p><strong>{i}. {stats.get('nome_polo', 'Polo')}:</strong>
                    {stats['num_estabelecimentos']} estabelecimentos com densidade de {stats['densidade']:.1f} rest/km²</p>
"""

    html += """
                </div>
                <div class="success-box">
                    <h3>⭐ Melhor Qualidade</h3>
"""

    top3_qualidade = sorted(estatisticas, key=lambda x: x['avg_rating'],
                            reverse=True)[:3]
    for i, stats in enumerate(top3_qualidade, 1):
        html += f"""
                    <p><strong>{i}. {stats.get('nome_polo', 'Polo')}:</strong>
                    Avaliação média de {stats['avg_rating']:.2f} ⭐</p>
"""

    html += f"""
                </div>
                <div class="warning-box">
                    <h3>🔸 Estabelecimentos Isolados</h3>
                    <p>Foram identificados <strong>{np.sum(clusters == -1)}</strong> estabelecimentos isolados
                    ({(np.sum(clusters == -1) / len(clusters) * 100):.1f}% do total).</p>
                    <p style="margin-top: 10px;">Estes estabelecimentos não fazem parte de nenhum polo gastronômico 
                    identificado e podem representar oportunidades de desenvolvimento comercial.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>🔬 Metodologia</h2>
                <div class="info-box">
                    <h3>Algoritmo HDBSCAN</h3>
                    <p><strong>HDBSCAN</strong> (Hierarchical Density-Based Spatial Clustering of Applications with Noise) 
                    é um algoritmo de clustering baseado em densidade que identifica automaticamente o número de clusters.</p>
                    <p style="margin-top: 15px;"><strong>Parâmetros utilizados:</strong></p>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li><strong>min_cluster_size:</strong> {MIN_CLUSTER_SIZE} (tamanho mínimo do cluster)</li>
                        <li><strong>min_samples:</strong> {MIN_SAMPLES} (amostras mínimas para densidade)</li>
                        <li><strong>metric:</strong> haversine (distância geográfica real - resultados em km)</li>
                    </ul>
                    <p style="margin-top: 15px;"><strong>Fonte dos dados:</strong> Google Places API (New)</p>
                    <p><strong>Área de busca:</strong> Fortaleza, CE (Grid {PONTOS_GRID}x{PONTOS_GRID})</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Análise de Polos Gastronômicos - Fortaleza/CE</strong></p>
            <p>Desenvolvido com Python + HDBSCAN + Google Places API</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Relatório gerado automaticamente em {timestamp}</p>
        </div>
    </div>
</body>
</html>
"""

    with open('analise_avancada/relatorio_completo.html', 'w',
              encoding='utf-8') as f:
        f.write(html)

    print("Relatório salvo em: analise_avancada/relatorio_completo.html")
    print()

    caminho_completo = os.path.abspath('analise_avancada/relatorio_completo.html')
    print("Abrindo relatório no navegador...")
    webbrowser.open('file://' + caminho_completo)

def criar_mapa_calor(lugares, centro_lat, centro_lng, clusters=None,
                     estatisticas=None,
                     lugares_validos=None,
                     nome_arquivo="mapa_calor_fortaleza_hdbscan.html"):
    """
    Cria um mapa interativo em Folium com visualização de calor (heatmap) dos
    estabelecimentos e, opcionalmente, sobreposição de polos gastronômicos
    identificados pelo HDBSCAN e estabelecimentos isolados.
    
    Args:
        lugares (list): Lista de dicionários com informações de
        estabelecimentos, incluindo localização e avaliação.
        centro_lat (float): Latitude central do mapa.
        centro_lng (float): Longitude central do mapa.
        clusters (numpy.ndarray, optional): Vetor de rótulos de cluster
        atribuídos a cada estabelecimento.
        estatisticas (list, optional): Lista de dicionários com métricas e 
        informações dos polos identificados
        (centroide, densidade, raio, avaliação).
        lugares_validos (list, optional): Lista de estabelecimentos
        correspondentes aos clusters, usada para plotar outliers.
        nome_arquivo (str, optional): Nome do arquivo HTML de saída.
        
    Returns:
        folium.Map: Objeto Folium do mapa gerado, permitindo visualização
        ou manipulação adicional.
    """


    cores_clusters = ['red', 'blue', 'green', 'purple', 'orange',
                      'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'pink']

    mapa = folium.Map(
        location=[centro_lat, centro_lng],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    pontos_calor = []

    for lugar in lugares:
        loc = lugar.get("location", {})
        lat = loc.get("latitude")
        lng = loc.get("longitude")

        if lat and lng:
            pontos_calor.append([lat, lng, 1])

    if pontos_calor:
        HeatMap(
            pontos_calor,
            radius=15,
            blur=25,
            max_zoom=13,
            gradient={
                0.0: 'blue',
                0.3: 'cyan',
                0.5: 'lime',
                0.7: 'yellow',
                1.0: 'red'
            }
        ).add_to(mapa)

    if estatisticas is not None and MOSTRAR_CLUSTERS:
        for idx, stats in enumerate(estatisticas):
            centroide = stats['centroide']
            cor = cores_clusters[idx % len(cores_clusters)]
            nome_polo = stats.get('nome_polo', f'Polo {idx+1}')

            folium.Circle(
                location=[centroide[0], centroide[1]],
                radius=stats['raio_km'] * 1000,
                color=cor,
                fill=True,
                fillColor=cor,
                fillOpacity=0.15,
                opacity=0.6,
                weight=2,
                popup=folium.Popup(f"""
                    <div style="width:240px">
                        <h3>🍽️ {nome_polo}</h3>
                        <p><strong>Estabelecimentos:
                            </strong> {stats['num_estabelecimentos']}</p>
                        <p><strong>Raio:
                            </strong> {stats['raio_km']:.2f} km</p>
                        <p><strong>Densidade:
                            </strong> {stats['densidade']:.1f} rest/km²</p>
                        <p><strong>Avaliação média:
                            </strong> {stats['avg_rating']:.2f} ⭐</p>
                    </div>
                """, max_width=260)
            ).add_to(mapa)

            folium.Marker(
                location=[centroide[0], centroide[1]],
                icon=folium.Icon(color=cor, icon='cutlery', prefix='fa'),
                popup=folium.Popup(f"""
                    <div style="width:260px">
                        <h3>🎯 {nome_polo}</h3>
                        <hr>
                        <p><strong>📊 Estabelecimentos:
                            </strong> {stats['num_estabelecimentos']} restaurantes</p>
                        <p><strong>📏 Área:
                            </strong> ~{stats['raio_km']:.2f} km de raio</p>
                        <p><strong>🔥 Concentração:
                            </strong> {stats['densidade']:.1f} rest/km²</p>
                        <p><strong>⭐ Qualidade:
                            </strong> {stats['avg_rating']:.2f}/5.0</p>
                        <p><strong>📍 Coordenadas:
                            </strong><br>
                           {centroide[0]:.4f}, {centroide[1]:.4f}</p>
                        <p style="color:gray; font-size:0.9em; margin-top:10px;">
                        Ordenado por densidade<br>
                        Identificado por HDBSCAN
                        </p>
                    </div>
                """, max_width=280),
                tooltip=f"{nome_polo}: {stats['num_estabelecimentos']} restaurantes"
            ).add_to(mapa)

    if clusters is not None and lugares_validos is not None and MOSTRAR_OUTLIERS:
        outlier_indices = np.where(clusters == -1)[0]
        if len(outlier_indices) > 0:
            for idx in outlier_indices:
                lugar = lugares_validos[idx]
                loc = lugar.get("location", {})
                lat = loc.get("latitude")
                lng = loc.get("longitude")
                nome = lugar.get("displayName", {}).get("text", "Sem nome")
                rating = lugar.get("rating", 0)

                if lat and lng:
                    folium.CircleMarker(
                        location=[lat, lng],
                        radius=4,
                        color='gray',
                        fill=True,
                        fillColor='lightgray',
                        fillOpacity=0.6,
                        weight=1,
                        popup=folium.Popup(f"""
                            <div style="width:180px">
                                <h4>🔸 Estabelecimento Isolado</h4>
                                <p><strong>{nome}</strong></p>
                                <p>⭐ {rating:.1f}</p>
                                <p style="color:gray; font-size:0.85em">
                                Não faz parte de um polo gastronômico
                                </p>
                            </div>
                        """, max_width=200),
                        tooltip=f"Isolado: {nome}"
                    ).add_to(mapa)

    try:
        caminho_completo = os.path.abspath(nome_arquivo)
        mapa.save(caminho_completo)

        if os.path.exists(caminho_completo):
            tamanho = os.path.getsize(caminho_completo) / 1024
            print("Mapa salvo com sucesso!")
            print(f"   Local: {caminho_completo}")
            print(f"   Tamanho: {tamanho:.1f} KB")
            print(f"   {len(pontos_calor)} pontos no mapa de calor")
            if estatisticas is not None:
                print(f"   {len(estatisticas)} polos gastronômicos identificados")
            if clusters is not None:
                num_outliers = np.sum(clusters == -1)
                print(f"   {num_outliers} estabelecimentos isolados")
            print()

            print("Abrindo mapa no navegador...")
            webbrowser.open('file://' + caminho_completo)
        else:
            print("Erro: O arquivo não foi criado")

    except Exception as e:
        print(f"Erro ao salvar: {e}")

    return mapa

def gerar_estatisticas(lugares):
    """
    Exibe estatísticas básicas dos estabelecimentos fornecidos, incluindo 
    total de estabelecimentos, avaliação média, melhor e pior avaliação.
    
    Args:
        lugares (list): Lista de dicionários com informações dos
        estabelecimentos, incluindo, opcionalmente, a chave "rating".
    
    Returns:
        None
    """
    print("ESTATÍSTICAS DOS ESTABELECIMENTOS")

    total = len(lugares)
    print(f"  Total de estabelecimentos: {total}")

    if total == 0:
        print("-" * 60)
        return

    avaliacoes = [p.get("rating", 0) for p in lugares if p.get("rating")]
    if avaliacoes:
        print(f"Avaliação média: {sum(avaliacoes)/len(avaliacoes):.2f}")
        print(f"Melhor: {max(avaliacoes):.1f}")
        print(f"Pior: {min(avaliacoes):.1f}")

    print("-" * 60)

def salvar_dados_para_streamlit(lugares, estatisticas,
                                clusters, lugares_validos):
    """
    Salva os dados de clustering e estatísticas dos polos gastronômicos em 
    arquivos para uso em um dashboard Streamlit.
    
    Args:
        lugares (list): Lista completa de estabelecimentos.
        estatisticas (list): Lista de dicionários com estatísticas de cada polo.
        clusters (array-like): Array com os rótulos de cluster para
        cada estabelecimento.
        lugares_validos (list): Subconjunto de estabelecimentos
        válidos utilizados no clustering.
        
    Arquivos gerados:
        - 'dados_clustering.pkl': Armazena todos os dados em formato pickle.
        - 'estatisticas_clusters.json': Estatísticas dos clusters em
        formato JSON.
    """
    print()
    print("SALVANDO DADOS PARA DASHBOARD STREAMLIT")

    dados = {
        'lugares': lugares,
        'estatisticas': estatisticas,
        'clusters': clusters,
        'lugares_validos': lugares_validos
    }

    try:
        with open('dados_clustering.pkl', 'wb') as f:
            pickle.dump(dados, f)
        print("Dados salvos em: dados_clustering.pkl")
    except Exception as e:
        print(f"Erro ao salvar pickle: {e}")

    try:
        estatisticas_json = []
        for stats in estatisticas:
            stats_copy = stats.copy()
            stats_copy['centroide'] = stats_copy['centroide'].tolist()
            stats_copy.pop('estabelecimentos', None)
            estatisticas_json.append(stats_copy)

        with open('estatisticas_clusters.json', 'w', encoding='utf-8') as f:
            json.dump(estatisticas_json, f, indent=2, ensure_ascii=False)
        print("Estatísticas salvas em: estatisticas_clusters.json")
    except Exception as e:
        print(f"Erro ao salvar JSON: {e}")

    print()
    print("Para visualizar no dashboard:")
    print("   Execute: streamlit run dashboard_streamlit.py")
    print()

if __name__ == "__main__":
    print("MAPA DE CALOR - FORTALEZA/CE + HDBSCAN")
    print("Versão com Análise Avançada Completa v3.0")
    print(f"Diretório: {os.getcwd()}")
    print()

    if API_KEY == "SUA_API_KEY_AQUI":
        print("  Configure sua API Key do Google Maps!")
        print("   Edite a variável API_KEY no início do código.")
    else:
        lugares = buscar_area_extensa(
            API_KEY,
            LATITUDE_MINIMA,
            LATITUDE_MAXIMA,
            LONGITUDE_MINIMA,
            LONGITUDE_MAXIMA,
            TIPOS_ESTABELECIMENTO,
            RAIO_BUSCA,
            PONTOS_GRID
        )

        if lugares:
            gerar_estatisticas(lugares)
            print()

            RESULTADOS_PARAMETROS = None
            if MODO_ANALISE_AVANCADA and TESTAR_PARAMETROS:
                RESULTADOS_PARAMETROS = testar_parametros_hdbscan(lugares)

            print("Aplicando HDBSCAN clustering...")
            resultado = identificar_polos_gastronomicos_hdbscan(
                lugares,
                MIN_CLUSTER_SIZE,
                MIN_SAMPLES
            )

            if resultado[1] is not None:
                clusters, estatisticas, lugares_validos = resultado
                print()

                METRICAS = None
                if MODO_ANALISE_AVANCADA:
                    print("Calculando métricas de qualidade...")

                    coords = np.array([[p.get("location", {}).get("latitude", 0),
                                       p.get("location",
                                             {}).get("longitude",
                                                     0)] for p in lugares_validos])

                    lat_ref = coords[0, 0]
                    coords_km = np.zeros_like(coords)

                    for i, (lat, lng) in enumerate(coords):
                        coords_km[i, 0] = haversine(lat_ref,
                                                    coords[0, 1],
                                                    lat, coords[0, 1])
                        coords_km[i, 1] = haversine(lat_ref,
                                                    coords[0, 1],
                                                    lat_ref, lng)

                        if lat < lat_ref:
                            coords_km[i, 0] *= -1
                        if lng < coords[0, 1]:
                            coords_km[i, 1] *= -1

                    METRICAS = calcular_METRICAS_qualidade(coords_km, clusters)

                    if METRICAS:
                        print()
                        print("MÉTRICAS DE QUALIDADE DO CLUSTERING")
                        print(f"   Silhouette Score: {METRICAS['silhouette']:.3f}")
                        print(f"   Davies-Bouldin Index: {METRICAS['davies_bouldin']:.3f}")
                        print(f"   Calinski-Harabasz Score: {METRICAS['calinski_harabasz']:.1f}")
                        print()

                if MODO_ANALISE_AVANCADA and GERAR_GRAFICOS:
                    gerar_graficos_analise(lugares, clusters,
                                           estatisticas, METRICAS,
                                           RESULTADOS_PARAMETROS)

                if MODO_ANALISE_AVANCADA and GERAR_RELATORIO:
                    gerar_relatorio_html(lugares, clusters,
                                         estatisticas, METRICAS,
                                         RESULTADOS_PARAMETROS)

                print("Gerando mapa de calor com polos gastronômicos...")
                criar_mapa_calor(
                    lugares,
                    AREA_CENTER["latitude"],
                    AREA_CENTER["longitude"],
                    clusters,
                    estatisticas,
                    lugares_validos
                )

                salvar_dados_para_streamlit(lugares, estatisticas,
                                            clusters, lugares_validos)

                print()
                print("ANÁLISE COMPLETA!")
                print()
                print("Arquivos gerados:")
                print("   • mapa_calor_fortaleza_hdbscan.html - Mapa interativo")
                print("   • dados_clustering.pkl - Dados para Streamlit")
                print("   • estatisticas_clusters.json - Estatísticas em JSON")

                if MODO_ANALISE_AVANCADA:
                    print()
                    print("Análise Avançada:")
                    if GERAR_GRAFICOS:
                        print("   -analise_avancada/01_distribuicao_clusters.png")
                        print("   -analise_avancada/02_densidade_vs_qualidade.png")
                        print("   -analise_avancada/03_distribuicao_avaliacoes.png")
                        print("   -analise_avancada/04_comparacao_METRICAS.png")
                        print("   -analise_avancada/05_dispersao_espacial.png")
                        if RESULTADOS_PARAMETROS:
                            print("   -analise_avancada/06_analise_parametros.png")
                    if GERAR_RELATORIO:
                        print("   -analise_avancada/relatorio_completo.html")

                print()
                print("Legenda do mapa:")
                print("   Círculos coloridos = Polos gastronômicos (HDBSCAN)")
                print("   Marcadores coloridos = Centro de cada polo")
                print("   Pontos cinzas = Estabelecimentos isolados")
                print("   Mapa de calor = Densidade geral")
                print()

            else:
                print("Não foi possível realizar o clustering")
        else:
            print("Nenhum estabelecimento encontrado.")
