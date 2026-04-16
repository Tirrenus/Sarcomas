#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 ANÁLISE DE ÓBITOS - DATASET DE SAÚDE PÚBLICA (2000-2019)
==========================================================

Código completo, modular e otimizado para processamento de dataset 
com mais de 22 milhões de linhas no Google Colab.

Autor: Especialista em Análise de Dados de Saúde Pública
Data: 2024
"""

# ============================================================================
# CONFIGURAÇÃO INICIAL E INSTALAÇÃO DE BIBLIOTECAS
# ============================================================================

# Instalar bibliotecas necessárias (descomente se necessário no Colab)
# !pip install pandas numpy matplotlib seaborn plotly polars

# Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import gc
from typing import Tuple, Dict, Optional, List
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Configurações de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configurar fonte para suportar caracteres especiais
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("✅ Bibliotecas importadas com sucesso!")


# ============================================================================
# CONFIGURAÇÕES GLOBAIS E CONSTANTES
# ============================================================================

# Faixas etárias padrão conforme especificação
AGE_BINS = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
            "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]

# Limites das faixas etárias
AGE_BIN_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 120]
AGE_BIN_LABELS = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
                  "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]

# Códigos CID de interesse para sarcomas
SARCOMA_CODES = ['C40', 'C41', 'C49']
BONE_SARCOMA_CODES = ['C40', 'C41']
SOFT_TISSUE_SARCOMA_CODES = ['C49']

# Códigos CID para neoplasmas
NEOPLASM_PREFIX = 'C'
NEOPLASM_D46 = 'D46'

# CIDs específicos para análise de jovens
YOUNG_CIDS = ['C40', 'C41', 'C49', 'C91', 'C95', 'C70', 'C72']
YOUNG_AGE_BINS = ["0-4", "5-9", "10-14", "15-19"]

# Cores para visualizações
COLOR_PALETTE = {
    'C40': '#FF6B6B',
    'C41': '#4ECDC4',
    'C49': '#45B7D1',
    'neoplasms': '#96CEB4',
    'all_causes': '#FFEAA7',
    'bone_sarcoma': '#DDA0DD',
    'soft_tissue': '#98FB98'
}


# ============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO (SEÇÃO 1.1)
# ============================================================================

def optimize_dtypes() -> Dict[str, str]:
    """
    Define tipos de dados otimizados para reduzir uso de memória.
    
    Returns:
        Dict com mapeamento de colunas para tipos de dados
    """
    return {
        'ano': 'int16',
        'sigla_uf': 'category',
        'causa_basica': 'string',
        'idade': 'float32',
        'sexo': 'category',
        'raca_cor': 'category'
    }


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nomes das colunas: lowercase e underscores.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame com nomes de colunas normalizados
    """
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('-', '_'))
    return df


def clean_string_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Remove espaços extras de colunas string.
    
    Args:
        df: DataFrame
        columns: Lista de colunas para limpar
        
    Returns:
        DataFrame com colunas limpas
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def group_cid_codes(cid: str) -> str:
    """
    Agrupa códigos CID conforme especificação:
    - C400-C409 → C40
    - C410-C419 → C41
    - C490-C499 → C49
    - Outros códigos C → primeiros 3 caracteres
    
    Args:
        cid: Código CID original
        
    Returns:
        Código CID agrupado
    """
    if pd.isna(cid) or not isinstance(cid, str):
        return np.nan
    
    cid = cid.strip().upper()
    
    # Para códigos que começam com C e têm 4+ caracteres
    if cid.startswith('C') and len(cid) >= 4:
        prefix = cid[:3]
        # Verifica se o quarto caractere é dígito
        if cid[3].isdigit():
            return prefix
    
    # Retorna os primeiros 3 caracteres para outros casos
    return cid[:3] if len(cid) >= 3 else cid


def create_cause_root_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria coluna 'causa_raiz' com códigos CID agrupados.
    
    Args:
        df: DataFrame com coluna 'causa_basica'
        
    Returns:
        DataFrame com nova coluna 'causa_raiz'
    """
    df['causa_raiz'] = df['causa_basica'].apply(group_cid_codes)
    return df


def categorize_age(age: float) -> Optional[str]:
    """
    Categoriza idade em faixas etárias predefinidas.
    
    Args:
        age: Idade em anos
        
    Returns:
        Faixa etária ou None se inválido
    """
    if pd.isna(age) or age < 0:
        return None
    
    for i, edge in enumerate(AGE_BIN_EDGES[:-1]):
        if age >= edge and age < AGE_BIN_EDGES[i + 1]:
            return AGE_BIN_LABELS[i]
    
    # Idades >= 80
    if age >= 80:
        return "80+"
    
    return None


def load_dataset_optimized(filepath: str, chunksize: int = 500000, 
                           sample_fraction: Optional[float] = None,
                           verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Carrega dataset grande de forma otimizada usando chunking.
    
    Args:
        filepath: Caminho para o arquivo CSV
        chunksize: Tamanho de cada chunk para leitura
        sample_fraction: Fração para amostragem (None = carrega tudo)
        verbose: Se True, imprime progresso
        
    Returns:
        Tuple com DataFrame limpo e estatísticas de qualidade
    """
    if verbose:
        print(f"📂 Carregando dataset: {filepath}")
        print(f"📊 Chunk size: {chunksize:,} linhas")
    
    chunks = []
    total_rows = 0
    quality_stats = {
        'total_rows_loaded': 0,
        'rows_with_null_cause': 0,
        'rows_with_null_age': 0,
        'rows_with_null_sex': 0,
        'rows_with_null_race': 0,
        'invalid_ages': 0,
        'chunks_processed': 0
    }
    
    # Leitura em chunks
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize)):
        # Normalizar nomes das colunas
        chunk = normalize_column_names(chunk)
        
        # Limpar espaços das colunas string
        string_cols = ['sigla_uf', 'causa_basica', 'sexo', 'raca_cor']
        chunk = clean_string_columns(chunk, string_cols)
        
        # Contabilizar estatísticas de qualidade
        quality_stats['total_rows_loaded'] += len(chunk)
        quality_stats['rows_with_null_cause'] += chunk['causa_basica'].isna().sum()
        quality_stats['rows_with_null_age'] += chunk['idade'].isna().sum()
        quality_stats['rows_with_null_sex'] += chunk['sexo'].isna().sum()
        quality_stats['rows_with_null_race'] += chunk['raca_cor'].isna().sum()
        
        # Converter idade para numérico
        chunk['idade'] = pd.to_numeric(chunk['idade'], errors='coerce')
        invalid_ages = (chunk['idade'] < 0).sum()
        quality_stats['invalid_ages'] += invalid_ages
        
        # Amostrar se necessário
        if sample_fraction is not None and sample_fraction < 1.0:
            chunk = chunk.sample(frac=sample_fraction, random_state=42)
        
        chunks.append(chunk)
        total_rows += len(chunk)
        quality_stats['chunks_processed'] += 1
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processados {i + 1} chunks ({total_rows:,} linhas)")
        
        # Liberar memória
        gc.collect()
    
    # Concatenar todos os chunks
    if verbose:
        print(f"\n🔄 Concatenando {len(chunks)} chunks...")
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Liberar memória dos chunks
    del chunks
    gc.collect()
    
    if verbose:
        print(f"✅ Dataset carregado: {len(df):,} linhas")
    
    return df, quality_stats


def preprocess_data(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Função completa de pré-processamento.
    
    Args:
        df: DataFrame bruto
        verbose: Se True, imprime informações
        
    Returns:
        Tuple com DataFrame processado e estatísticas
    """
    stats = {'initial_rows': len(df), 'operations': []}
    
    if verbose:
        print("\n🔧 Iniciando pré-processamento...")
    
    # 1. Normalizar nomes das colunas
    df = normalize_column_names(df)
    stats['operations'].append('Normalização de colunas')
    
    # 2. Limpar strings
    string_cols = ['sigla_uf', 'causa_basica', 'sexo', 'raca_cor']
    df = clean_string_columns(df, string_cols)
    stats['operations'].append('Limpeza de strings')
    
    # 3. Converter idade para numérico
    df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
    stats['operations'].append('Conversão de idade')
    
    # 4. Criar coluna causa_raiz
    df = create_cause_root_column(df)
    stats['operations'].append('Criação de causa_raiz')
    
    # 5. Criar faixa etária
    df['faixa_etaria'] = df['idade'].apply(categorize_age)
    stats['operations'].append('Categorização de idade')
    
    # 6. Remover valores NaN críticos
    null_before = len(df)
    df = df.dropna(subset=['causa_basica'])
    stats['removed_null_cause'] = null_before - len(df)
    stats['operations'].append('Remoção de causas nulas')
    
    stats['final_rows'] = len(df)
    
    if verbose:
        print(f"✅ Pré-processamento concluído!")
        print(f"   Linhas iniciais: {stats['initial_rows']:,}")
        print(f"   Linhas finais: {stats['final_rows']:,}")
        print(f"   Removidas: {stats.get('removed_null_cause', 0):,}")
    
    return df, stats


# ============================================================================
# FUNÇÕES DE EXPLORAÇÃO DE DADOS (SEÇÃO 1.2)
# ============================================================================

def generate_eda_report(df: pd.DataFrame, save_path: Optional[str] = None) -> Dict:
    """
    Gera relatório exploratório completo do dataset.
    
    Args:
        df: DataFrame processado
        save_path: Caminho para salvar relatório (opcional)
        
    Returns:
        Dict com estatísticas do EDA
    """
    print("\n" + "="*80)
    print("📊 ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("="*80)
    
    eda_stats = {}
    
    # 1. Informações básicas
    print("\n--- INFO() ---")
    print(f"Shape: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    eda_stats['shape'] = df.shape
    
    # 2. Describe
    print("\n--- DESCRIBE() ---")
    print(df.describe())
    eda_stats['describe'] = df.describe().to_dict()
    
    # 3. Distribuição de nulos
    print("\n--- DISTRIBUIÇÃO DE NULOS ---")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    null_df = pd.DataFrame({
        'Nulos': null_counts,
        'Percentual (%)': null_pct
    })
    print(null_df)
    eda_stats['nulls'] = null_df.to_dict()
    
    # 4. Top 20 causas básicas
    print("\n--- TOP 20 CAUSAS BÁSICAS ---")
    top_causes = df['causa_basica'].value_counts().head(20)
    print(top_causes)
    eda_stats['top_causes'] = top_causes.to_dict()
    
    # 5. Distribuição por sexo
    print("\n--- DISTRIBUIÇÃO POR SEXO ---")
    sex_dist = df['sexo'].value_counts()
    print(sex_dist)
    eda_stats['sex_distribution'] = sex_dist.to_dict()
    
    # 6. Distribuição por raça/cor
    print("\n--- DISTRIBUIÇÃO POR RAÇA/COR ---")
    race_dist = df['raca_cor'].value_counts()
    print(race_dist)
    eda_stats['race_distribution'] = race_dist.to_dict()
    
    # 7. Distribuição por UF
    print("\n--- DISTRIBUIÇÃO POR UF ---")
    uf_dist = df['sigla_uf'].value_counts()
    print(uf_dist)
    eda_stats['uf_distribution'] = uf_dist.to_dict()
    
    # 8. Estatísticas de idade
    print("\n--- ESTATÍSTICAS DE IDADE ---")
    age_stats = df['idade'].describe()
    print(age_stats)
    eda_stats['age_statistics'] = age_stats.to_dict()
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO EDA\n")
            f.write("="*80 + "\n\n")
            f.write(f"Shape: {df.shape}\n\n")
            f.write("DESCRIBE:\n")
            f.write(df.describe().to_string() + "\n\n")
            f.write("NULOS:\n")
            f.write(null_df.to_string() + "\n\n")
            f.write("TOP CAUSAS:\n")
            f.write(top_causes.to_string() + "\n")
    
    return eda_stats


def plot_age_histogram(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plota histograma de idades.
    
    Args:
        df: DataFrame
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    valid_ages = df['idade'].dropna()
    ax.hist(valid_ages, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Idade', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    ax.set_title('Distribuição de Idades nos Óbitos', fontsize=14, fontweight='bold')
    ax.axvline(valid_ages.median(), color='red', linestyle='--', 
               label=f'Mediana: {valid_ages.median():.1f} anos')
    ax.legend()
    plt.tight_layout()
    
    return fig


def plot_categorical_distribution(df: pd.DataFrame, column: str, 
                                   top_n: int = 15,
                                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plota distribuição de variável categórica.
    
    Args:
        df: DataFrame
        column: Nome da coluna
        top_n: Número de categorias para mostrar
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    counts = df[column].value_counts().head(top_n)
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    
    bars = ax.bar(range(len(counts)), counts.values, color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha='right')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Contagem', fontsize=12)
    ax.set_title(f'Distribuição de {column} (Top {top_n})', fontsize=14, fontweight='bold')
    
    # Adicionar valores nas barras
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts.values)*0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO REUTILIZÁVEIS
# ============================================================================

def create_stacked_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, 
                              hue_col: str, title: str,
                              xlabel: str, ylabel: str,
                              figsize: Tuple[int, int] = (14, 8),
                              palette: Optional[Dict] = None,
                              log_scale: bool = False) -> plt.Figure:
    """
    Cria gráfico de barras empilhadas.
    
    Args:
        data: DataFrame
        x_col: Coluna para eixo X
        y_col: Coluna para eixo Y
        hue_col: Coluna para agrupamento (cores)
        title: Título do gráfico
        xlabel: Label do eixo X
        ylabel: Label do eixo Y
        figsize: Tamanho da figura
        palette: Dicionário de cores
        log_scale: Se True, usa escala logarítmica
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivotar dados para barras empilhadas
    pivot_data = data.pivot(index=x_col, columns=hue_col, values=y_col)
    
    # Ordenar por faixa etária
    if x_col == 'faixa_etaria':
        pivot_data = pivot_data.reindex(AGE_BINS)
    
    # Plotar
    pivot_data.plot(kind='bar', stacked=True, ax=ax, 
                    colormap='Set2' if palette is None else None,
                    width=0.8)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    return fig


def create_grouped_bar_chart(data: pd.DataFrame, x_col: str, y_col: str,
                              hue_col: str, title: str,
                              xlabel: str, ylabel: str,
                              figsize: Tuple[int, int] = (14, 8),
                              palette: Optional[List[str]] = None) -> plt.Figure:
    """
    Cria gráfico de barras agrupadas.
    
    Args:
        data: DataFrame
        x_col: Coluna para eixo X
        y_col: Coluna para eixo Y
        hue_col: Coluna para agrupamento
        title: Título
        xlabel: Label X
        ylabel: Label Y
        figsize: Tamanho da figura
        palette: Lista de cores
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivotar dados
    pivot_data = data.pivot(index=x_col, columns=hue_col, values=y_col)
    
    # Ordenar se for faixa etária
    if x_col == 'faixa_etaria':
        pivot_data = pivot_data.reindex(AGE_BINS)
    
    # Plotar barras agrupadas
    pivot_data.plot(kind='bar', ax=ax, width=0.8, 
                    colormap='Set1' if palette is None else None)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def create_simple_bar_chart(data: pd.Series, title: str, xlabel: str, ylabel: str,
                            figsize: Tuple[int, int] = (14, 8),
                            color: str = 'steelblue') -> plt.Figure:
    """
    Cria gráfico de barras simples.
    
    Args:
        data: Series com dados
        title: Título
        xlabel: Label X
        ylabel: Label Y
        figsize: Tamanho da figura
        color: Cor das barras
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ordenar se for faixa etária
    if data.index.name == 'faixa_etaria' or list(data.index)[:1] in [AGE_BINS[0]]:
        data = data.reindex([idx for idx in AGE_BINS if idx in data.index])
    
    bars = ax.bar(range(len(data)), data.values, color=color, edgecolor='black')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.index, rotation=45, ha='right')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Adicionar valores
    for bar, val in zip(bars, data.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data.values)*0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_line_chart(data: pd.DataFrame, x_col: str, y_col: str,
                      hue_col: Optional[str] = None, title: str = '',
                      xlabel: str = '', ylabel: str = '',
                      figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Cria gráfico de linhas.
    
    Args:
        data: DataFrame
        x_col: Coluna X
        y_col: Coluna Y
        hue_col: Coluna para múltiplas linhas
        title: Título
        xlabel: Label X
        ylabel: Label Y
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if hue_col is None:
        # Linha única
        ax.plot(data[x_col], data[y_col], marker='o', linewidth=2, markersize=6)
    else:
        # Múltiplas linhas
        pivot_data = data.pivot(index=x_col, columns=hue_col, values=y_col)
        if x_col == 'faixa_etaria':
            pivot_data = pivot_data.reindex(AGE_BINS)
        
        for col in pivot_data.columns:
            ax.plot(pivot_data.index, pivot_data[col], marker='o', 
                   linewidth=2, markersize=6, label=col)
        ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_horizontal_bar_chart(data: pd.DataFrame, value_col: str, 
                                 index_col: str, title: str,
                                 xlabel: str, ylabel: str,
                                 figsize: Tuple[int, int] = (10, 8),
                                 color: str = 'steelblue') -> plt.Figure:
    """
    Cria gráfico de barras horizontais.
    
    Args:
        data: DataFrame
        value_col: Coluna de valores
        index_col: Coluna de índices
        title: Título
        xlabel: Label X
        ylabel: Label Y
        figsize: Tamanho da figura
        color: Cor das barras
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(data[index_col], data[value_col], color=color)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Adicionar valores
    for bar, val in zip(bars, data[value_col]):
        ax.text(val + max(data[value_col])*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:,.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


# ============================================================================
# FUNÇÕES PARA INTERATIVIDADE COM PLOTLY
# ============================================================================

def create_interactive_bar_chart(df: pd.DataFrame, x: str, y: str, 
                                  color: Optional[str] = None,
                                  title: str = '', log_y: bool = False) -> go.Figure:
    """
    Cria gráfico de barras interativo com Plotly.
    
    Args:
        df: DataFrame
        x: Coluna X
        y: Coluna Y
        color: Coluna para cores
        title: Título
        log_y: Se True, usa escala logarítmica no Y
        
    Returns:
        Figura Plotly
    """
    fig = px.bar(df, x=x, y=y, color=color, title=title,
                 labels={x: x, y: y},
                 log_y=log_y)
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(tickangle=45)
    return fig


def create_interactive_line_chart(df: pd.DataFrame, x: str, y: str,
                                   color: Optional[str] = None,
                                   title: str = '') -> go.Figure:
    """
    Cria gráfico de linhas interativo com Plotly.
    
    Args:
        df: DataFrame
        x: Coluna X
        y: Coluna Y
        color: Coluna para cores
        title: Título
        
    Returns:
        Figura Plotly
    """
    fig = px.line(df, x=x, y=y, color=color, title=title,
                  markers=True)
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(tickangle=45)
    return fig


# ============================================================================
# ANÁLISES ESPECÍFICAS (SEÇÕES 2.x, 4.x, 5.x)
# ============================================================================

def analyze_sarcomas_by_age(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 2.1 - Óbitos por Sarcomas (C40, C41, C49) por faixa etária.
    
    Args:
        df: DataFrame processado
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 2.1 - Óbitos por Sarcomas (C40, C41, C49) por Faixa Etária")
    print("="*80)
    
    # Filtrar sarcomas
    sarcoma_df = df[df['causa_raiz'].isin(SARCOMA_CODES)].copy()
    sarcoma_df = sarcoma_df[sarcoma_df['faixa_etaria'].notna()]
    
    # Agrupar por causa_raiz e faixa_etaria
    grouped = sarcoma_df.groupby(['causa_raiz', 'faixa_etaria']).size().reset_index(name='count')
    
    # Criar tabela pivô
    pivot_table = grouped.pivot(index='faixa_etaria', columns='causa_raiz', values='count').fillna(0)
    pivot_table = pivot_table.reindex(AGE_BINS)
    
    print("\n📋 TABELA: Óbitos por Sarcomas por Faixa Etária")
    print(pivot_table.to_string())
    
    # Gráfico
    fig = create_grouped_bar_chart(
        grouped, 
        x_col='faixa_etaria', 
        y_col='count', 
        hue_col='causa_raiz',
        title='Number of deaths from sarcomas by Age Group',
        xlabel='Age Group',
        ylabel='Number of deaths from sarcomas (overall)',
        figsize=(14, 8)
    )
    
    return pivot_table, fig


def analyze_all_neoplasms_by_age(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 2.2 - Óbitos por todos os neoplasmas (C00-C97, D46) por faixa etária.
    
    Args:
        df: DataFrame processado
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 2.2 - Óbitos por Todos os Neoplasmas (C00-C97, D46) por Faixa Etária")
    print("="*80)
    
    # Filtrar neoplasmas: causa_basica começando com C OU igual D46
    neoplasm_df = df[(df['causa_basica'].str.startswith('C', na=False)) | 
                     (df['causa_basica'] == 'D46')].copy()
    neoplasm_df = neoplasm_df[neoplasm_df['faixa_etaria'].notna()]
    
    # Agrupar por faixa_etaria
    grouped = neoplasm_df.groupby('faixa_etaria').size().reset_index(name='count')
    grouped = grouped.set_index('faixa_etaria').reindex(AGE_BINS).fillna(0).reset_index()
    
    print("\n📋 TABELA: Óbitos por Neoplasmas por Faixa Etária")
    print(grouped.to_string(index=False))
    
    # Gráfico
    fig = create_simple_bar_chart(
        grouped.set_index('faixa_etaria')['count'],
        title='Number of deaths from all neoplasms by Age Group',
        xlabel='Age Group',
        ylabel='Number of deaths from all neoplasms',
        figsize=(14, 8),
        color=COLOR_PALETTE['neoplasms']
    )
    
    return grouped, fig


def analyze_all_causes_by_age(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 2.3 - Óbitos por todas as causas por faixa etária.
    
    Args:
        df: DataFrame processado
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 2.3 - Óbitos por Todas as Causas por Faixa Etária")
    print("="*80)
    
    # Considerar todo o dataset
    all_df = df[df['faixa_etaria'].notna()].copy()
    
    # Agrupar por faixa_etaria
    grouped = all_df.groupby('faixa_etaria').size().reset_index(name='count')
    grouped = grouped.set_index('faixa_etaria').reindex(AGE_BINS).fillna(0).reset_index()
    
    print("\n📋 TABELA: Óbitos por Todas as Causas por Faixa Etária")
    print(grouped.to_string(index=False))
    
    # Gráfico
    fig = create_simple_bar_chart(
        grouped.set_index('faixa_etaria')['count'],
        title='Number of deaths from all causes by Age Group',
        xlabel='Age Group',
        ylabel='Number of deaths from all causes',
        figsize=(14, 8),
        color=COLOR_PALETTE['all_causes']
    )
    
    return grouped, fig


def analyze_bone_sarcoma_percentage(df: pd.DataFrame, 
                                     total_deaths_by_age: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 2.4 - Percentual de óbitos por Sarcomas Ósseos (C40, C41).
    
    Args:
        df: DataFrame processado
        total_deaths_by_age: DataFrame com total de óbitos por faixa etária
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 2.4 - Percentual de Óbitos por Sarcomas Ósseos (C40, C41)")
    print("="*80)
    
    # Filtrar sarcomas ósseos
    bone_sarcoma_df = df[df['causa_raiz'].isin(BONE_SARCOMA_CODES)].copy()
    bone_sarcoma_df = bone_sarcoma_df[bone_sarcoma_df['faixa_etaria'].notna()]
    
    # Agrupar por faixa_etaria
    bone_grouped = bone_sarcoma_df.groupby('faixa_etaria').size().reset_index(name='bone_count')
    bone_grouped = bone_grouped.set_index('faixa_etaria').reindex(AGE_BINS).fillna(0).reset_index()
    
    # Calcular percentual
    merged = bone_grouped.merge(total_deaths_by_age, on='faixa_etaria', suffixes=('_bone', '_total'))
    merged['percentage'] = (merged['bone_count'] / merged['count'] * 100).round(4)
    
    print("\n📋 TABELA: Percentual de Óbitos por Sarcomas Ósseos")
    print(merged[['faixa_etaria', 'bone_count', 'count', 'percentage']].to_string(index=False))
    
    # Gráfico de linhas mostrando percentual
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(merged['faixa_etaria'], merged['percentage'], marker='o', 
            linewidth=2, markersize=8, color=COLOR_PALETTE['bone_sarcoma'])
    ax.fill_between(merged['faixa_etaria'], merged['percentage'], alpha=0.3, 
                    color=COLOR_PALETTE['bone_sarcoma'])
    
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Percentage of deaths from bone sarcomas by Age Group', 
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Adicionar valores
    for i, row in merged.iterrows():
        ax.annotate(f'{row["percentage"]:.3f}%', (row['faixa_etaria'], row['percentage']),
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    
    plt.tight_layout()
    return merged, fig


def analyze_soft_tissue_sarcoma_percentage(df: pd.DataFrame,
                                            total_deaths_by_age: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 2.5 - Percentual de óbitos por Sarcomas de Tecidos Moles (C49).
    
    Args:
        df: DataFrame processado
        total_deaths_by_age: DataFrame com total de óbitos por faixa etária
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 2.5 - Percentual de Óbitos por Sarcomas de Tecidos Moles (C49)")
    print("="*80)
    
    # Filtrar sarcomas de tecidos moles
    soft_df = df[df['causa_raiz'] == 'C49'].copy()
    soft_df = soft_df[soft_df['faixa_etaria'].notna()]
    
    # Agrupar por faixa_etaria
    soft_grouped = soft_df.groupby('faixa_etaria').size().reset_index(name='soft_count')
    soft_grouped = soft_grouped.set_index('faixa_etaria').reindex(AGE_BINS).fillna(0).reset_index()
    
    # Calcular percentual
    merged = soft_grouped.merge(total_deaths_by_age, on='faixa_etaria', suffixes=('_soft', '_total'))
    merged['percentage'] = (merged['soft_count'] / merged['count'] * 100).round(4)
    
    print("\n📋 TABELA: Percentual de Óbitos por Sarcomas de Tecidos Moles")
    print(merged[['faixa_etaria', 'soft_count', 'count', 'percentage']].to_string(index=False))
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(merged['faixa_etaria'], merged['percentage'], marker='s', 
            linewidth=2, markersize=8, color=COLOR_PALETTE['soft_tissue'])
    ax.fill_between(merged['faixa_etaria'], merged['percentage'], alpha=0.3, 
                    color=COLOR_PALETTE['soft_tissue'])
    
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Percentage of deaths from soft tissue sarcomas by Age Group', 
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Adicionar valores
    for i, row in merged.iterrows():
        ax.annotate(f'{row["percentage"]:.3f}%', (row['faixa_etaria'], row['percentage']),
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    
    plt.tight_layout()
    return merged, fig


def analyze_sarcomas_by_race(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 4.1 - Sarcomas (C40, C41, C49) por raça.
    
    Args:
        df: DataFrame processado
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 4.1 - Sarcomas (C40, C41, C49) por Raça")
    print("="*80)
    
    # Filtrar sarcomas
    sarcoma_df = df[df['causa_raiz'].isin(SARCOMA_CODES)].copy()
    
    # Agrupar por causa_raiz e raca_cor
    grouped = sarcoma_df.groupby(['causa_raiz', 'raca_cor']).size().reset_index(name='count')
    
    # Tabela pivô
    pivot_table = grouped.pivot(index='raca_cor', columns='causa_raiz', values='count').fillna(0)
    
    print("\n📋 TABELA: Sarcomas por Raça")
    print(pivot_table.to_string())
    
    # Gráfico de barras agrupadas
    fig = create_grouped_bar_chart(
        grouped,
        x_col='raca_cor',
        y_col='count',
        hue_col='causa_raiz',
        title='Sarcomas Deaths by Race/Color',
        xlabel='Race/Color',
        ylabel='Number of deaths',
        figsize=(12, 8)
    )
    
    return pivot_table, fig


def analyze_all_cancer_by_race(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 4.2 - Câncer Total (C00-C97 + D46) por raça.
    
    Args:
        df: DataFrame processado
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 4.2 - Câncer Total (C00-C97 + D46) por Raça")
    print("="*80)
    
    # Filtrar cânceres
    cancer_df = df[(df['causa_basica'].str.startswith('C', na=False)) | 
                   (df['causa_basica'] == 'D46')].copy()
    
    # Agrupar por raca_cor
    grouped = cancer_df.groupby('raca_cor').size().reset_index(name='count')
    total = grouped['count'].sum()
    grouped['percentage'] = (grouped['count'] / total * 100).round(2)
    
    print("\n📋 TABELA: Câncer Total por Raça")
    print(grouped.to_string(index=False))
    
    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(grouped['raca_cor'], grouped['count'], 
                  color=plt.cm.Pastel1(np.linspace(0, 1, len(grouped))))
    ax.set_xlabel('Race/Color', fontsize=12)
    ax.set_ylabel('Number of deaths', fontsize=12)
    ax.set_title('Total Cancer Deaths by Race/Color', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Adicionar valores e percentuais
    for bar, row in zip(bars, grouped.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(grouped['count'])*0.01,
                f'{row.count:,}\n({row.percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return grouped, fig


def analyze_bone_sarcomas_by_race(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 4.3a - Sarcomas Ósseos (C40, C41) por raça.
    
    Args:
        df: DataFrame processado
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 4.3a - Sarcomas Ósseos (C40, C41) por Raça")
    print("="*80)
    
    # Filtrar sarcomas ósseos
    bone_df = df[df['causa_raiz'].isin(BONE_SARCOMA_CODES)].copy()
    
    # Agrupar por causa_raiz e raca_cor
    grouped = bone_df.groupby(['causa_raiz', 'raca_cor']).size().reset_index(name='count')
    
    # Tabela pivô
    pivot_table = grouped.pivot(index='raca_cor', columns='causa_raiz', values='count').fillna(0)
    
    print("\n📋 TABELA: Sarcomas Ósseos por Raça")
    print(pivot_table.to_string())
    
    # Gráfico
    fig = create_grouped_bar_chart(
        grouped,
        x_col='raca_cor',
        y_col='count',
        hue_col='causa_raiz',
        title='Bone Sarcomas Deaths by Race/Color',
        xlabel='Race/Color',
        ylabel='Number of deaths',
        figsize=(12, 8)
    )
    
    return pivot_table, fig


def analyze_soft_tissue_sarcomas_by_race(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 4.3b - Sarcomas Tecidos Moles (C49) por raça.
    
    Args:
        df: DataFrame processado
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 4.3b - Sarcomas Tecidos Moles (C49) por Raça")
    print("="*80)
    
    # Filtrar sarcomas de tecidos moles
    soft_df = df[df['causa_raiz'] == 'C49'].copy()
    
    # Agrupar por raca_cor
    grouped = soft_df.groupby('raca_cor').size().reset_index(name='count')
    
    print("\n📋 TABELA: Sarcomas de Tecidos Moles por Raça")
    print(grouped.to_string(index=False))
    
    # Gráfico
    fig = create_simple_bar_chart(
        grouped.set_index('raca_cor')['count'],
        title='Soft Tissue Sarcomas Deaths by Race/Color',
        xlabel='Race/Color',
        ylabel='Number of deaths',
        figsize=(12, 8),
        color=COLOR_PALETTE['soft_tissue']
    )
    
    return grouped, fig


def analyze_specific_cids_young(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 5.1 - Comparação de CIDs específicos em jovens (0-14 e 15-19 anos).
    
    Args:
        df: DataFrame processado
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 5.1 - CIDs Específicos em Jovens (0-19 anos)")
    print("="*80)
    
    # Filtrar CIDs de interesse e faixas etárias jovens
    young_df = df[(df['causa_raiz'].isin(YOUNG_CIDS)) & 
                  (df['faixa_etaria'].isin(YOUNG_AGE_BINS))].copy()
    
    # Agrupar por causa_raiz e faixa_etaria
    grouped = young_df.groupby(['causa_raiz', 'faixa_etaria']).size().reset_index(name='count')
    
    # Tabela pivô
    pivot_table = grouped.pivot(index='causa_raiz', columns='faixa_etaria', values='count').fillna(0)
    pivot_table = pivot_table.reindex(YOUNG_CIDS)
    pivot_table = pivot_table[[col for col in YOUNG_AGE_BINS if col in pivot_table.columns]]
    
    print("\n📋 TABELA: CIDs Específicos em Jovens por Faixa Etária")
    print(pivot_table.to_string())
    
    # Gráfico de barras agrupadas
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Preparar dados para plot
    pivot_for_plot = pivot_table.reset_index()
    x = np.arange(len(pivot_for_plot))
    width = 0.2
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(YOUNG_AGE_BINS)))
    
    for i, age_bin in enumerate(YOUNG_AGE_BINS):
        if age_bin in pivot_for_plot.columns:
            ax.bar(x + i*width - width*(len(YOUNG_AGE_BINS)-1)/2, 
                   pivot_for_plot[age_bin], width, 
                   label=age_bin, color=colors[i])
    
    ax.set_xlabel('CID Code', fontsize=12)
    ax.set_ylabel('Number of deaths', fontsize=12)
    ax.set_title('Specific CIDs in Young People (0-19 years) by Age Group', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_for_plot['causa_raiz'])
    ax.legend(title='Age Group')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return pivot_table, fig


def calculate_asmr(deaths_by_age: pd.DataFrame, 
                   population_data: Optional[Dict[str, int]] = None,
                   use_relative_rate: bool = True) -> pd.DataFrame:
    """
    Calcula Age-Specific Mortality Rate (ASMR).
    
    ASMR = (Óbitos na faixa etária / População da faixa etária) * 100.000
    
    Args:
        deaths_by_age: DataFrame com óbitos por faixa etária
        population_data: Dicionário com população por faixa etária (opcional)
        use_relative_rate: Se True e sem population_data, usa taxa relativa
        
    Returns:
        DataFrame com ASMR calculado
    """
    result = deaths_by_age.copy()
    
    if population_data is not None:
        # Calcular ASMR com dados de população reais
        result['population'] = result['faixa_etaria'].map(population_data)
        result['asmr'] = (result['count'] / result['population'] * 100000).round(2)
    elif use_relative_rate:
        # Calcular taxa relativa usando total de óbitos como denominador
        total_deaths = result['count'].sum()
        result['asmr'] = (result['count'] / total_deaths * 100000).round(2)
        print("⚠️  Nota: Usando taxa relativa (óbitos/total óbitos * 100.000)")
        print("   Para ASMR real, forneça dados de população externa.")
    
    return result


def analyze_asmr_comparative(df: pd.DataFrame, 
                             population_data: Optional[Dict[str, int]] = None) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Seção 5.4 - Age-Specific Mortality Rate (ASMR) comparativo.
    
    Compara:
    - A = All neoplasms (C00-C97 + D46)
    - B = Sarcomas overall (C40, C41, C49)
    - C = Bone Sarcomas (C40, C41)
    - D = Soft tissue sarcomas (C49)
    
    Args:
        df: DataFrame processado
        population_data: Dados de população externos (opcional)
        
    Returns:
        Tuple com tabela de dados e figura
    """
    print("\n" + "="*80)
    print("📊 SEÇÃO 5.4 - Age-Specific Mortality Rate (ASMR) Comparativo")
    print("="*80)
    
    results = []
    
    # A = All neoplasms
    neoplasm_df = df[(df['causa_basica'].str.startswith('C', na=False)) | 
                     (df['causa_basica'] == 'D46')].copy()
    neoplasm_df = neoplasm_df[neoplasm_df['faixa_etaria'].notna()]
    neoplasm_by_age = neoplasm_df.groupby('faixa_etaria').size().reset_index(name='count')
    neoplasm_by_age = neoplasm_by_age.set_index('faixa_etaria').reindex(AGE_BINS).fillna(0).reset_index()
    neoplasm_by_age['category'] = 'A - All Neoplasms'
    results.append(neoplasm_by_age)
    
    # B = Sarcomas overall
    sarcoma_df = df[df['causa_raiz'].isin(SARCOMA_CODES)].copy()
    sarcoma_df = sarcoma_df[sarcoma_df['faixa_etaria'].notna()]
    sarcoma_by_age = sarcoma_df.groupby('faixa_etaria').size().reset_index(name='count')
    sarcoma_by_age = sarcoma_by_age.set_index('faixa_etaria').reindex(AGE_BINS).fillna(0).reset_index()
    sarcoma_by_age['category'] = 'B - Sarcomas Overall'
    results.append(sarcoma_by_age)
    
    # C = Bone Sarcomas
    bone_df = df[df['causa_raiz'].isin(BONE_SARCOMA_CODES)].copy()
    bone_df = bone_df[bone_df['faixa_etaria'].notna()]
    bone_by_age = bone_df.groupby('faixa_etaria').size().reset_index(name='count')
    bone_by_age = bone_by_age.set_index('faixa_etaria').reindex(AGE_BINS).fillna(0).reset_index()
    bone_by_age['category'] = 'C - Bone Sarcomas'
    results.append(bone_by_age)
    
    # D = Soft tissue sarcomas
    soft_df = df[df['causa_raiz'] == 'C49'].copy()
    soft_df = soft_df[soft_df['faixa_etaria'].notna()]
    soft_by_age = soft_df.groupby('faixa_etaria').size().reset_index(name='count')
    soft_by_age = soft_by_age.set_index('faixa_etaria').reindex(AGE_BINS).fillna(0).reset_index()
    soft_by_age['category'] = 'D - Soft Tissue Sarcomas'
    results.append(soft_by_age)
    
    # Combinar todos
    combined = pd.concat(results, ignore_index=True)
    
    # Calcular ASMR para cada categoria
    asmrs = []
    for category in combined['category'].unique():
        cat_data = combined[combined['category'] == category].copy()
        cat_asmr = calculate_asmr(cat_data, population_data)
        asmrs.append(cat_asmr)
    
    final_df = pd.concat(asmrs, ignore_index=True)
    
    print("\n📋 TABELA: ASMR por Categoria e Faixa Etária")
    print(final_df[['category', 'faixa_etaria', 'count', 'asmr']].to_string(index=False))
    
    # Gráfico de barras horizontais
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Preparar dados para gráfico
    pivot_asmr = final_df.pivot(index='faixa_etaria', columns='category', values='asmr')
    pivot_asmr = pivot_asmr.reindex(AGE_BINS)
    
    # Plotar barras horizontais para cada categoria
    y_pos = np.arange(len(AGE_BINS))
    bar_height = 0.2
    colors = {'A - All Neoplasms': '#FF6B6B',
              'B - Sarcomas Overall': '#4ECDC4',
              'C - Bone Sarcomas': '#45B7D1',
              'D - Soft Tissue Sarcomas': '#96CEB4'}
    
    for i, category in enumerate(['A - All Neoplasms', 'B - Sarcomas Overall', 
                                   'C - Bone Sarcomas', 'D - Soft Tissue Sarcomas']):
        offset = (i - 1.5) * bar_height
        values = pivot_asmr[category].values
        ax.barh(y_pos + offset, values, bar_height, label=category, color=colors[category])
    
    ax.set_xlabel('Age-Specific Mortality Rate (ASMR) per 100,000', fontsize=12)
    ax.set_ylabel('Age Group', fontsize=12)
    ax.set_title('ASMR Comparative Analysis by Category and Age Group', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(AGE_BINS)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Nota sobre limitação
    if population_data is None:
        print("\n⚠️  LIMITAÇÃO: Dados de população não fornecidos.")
        print("   ASMR calculado como taxa relativa (óbitos/total * 100.000).")
        print("   Para inserir dados de população, use o parâmetro population_data:")
        print("   Exemplo: population_data = {'0-4': 1000000, '5-9': 950000, ...}")
    
    return final_df, fig


# ============================================================================
# FUNÇÃO PRINCIPAL DE EXECUÇÃO
# ============================================================================

def run_full_analysis(filepath: str, sample_fraction: Optional[float] = None,
                      output_dir: str = 'output',
                      population_data: Optional[Dict[str, int]] = None) -> Dict:
    """
    Executa análise completa do dataset.
    
    Args:
        filepath: Caminho para o arquivo CSV
        sample_fraction: Fração para amostragem (None = carrega tudo)
        output_dir: Diretório para salvar outputs
        population_data: Dados de população para ASMR
        
    Returns:
        Dict com resultados da análise
    """
    print("\n" + "="*80)
    print("🏥 ANÁLISE DE ÓBITOS - SAÚDE PÚBLICA (2000-2019)")
    print("="*80)
    
    # Criar diretório de output
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # 1.1 - PRÉ-PROCESSAMENTO
    # =========================================================================
    print("\n" + "="*80)
    print("📋 SEÇÃO 1.1 - PRÉ-PROCESSAMENTO")
    print("="*80)
    
    df, load_stats = load_dataset_optimized(filepath, sample_fraction=sample_fraction)
    df, preprocess_stats = preprocess_data(df)
    
    results['load_stats'] = load_stats
    results['preprocess_stats'] = preprocess_stats
    
    # =========================================================================
    # 1.2 - ANÁLISE EXPLORATÓRIA (EDA)
    # =========================================================================
    eda_stats = generate_eda_report(df, save_path=f'{output_dir}/eda_report.txt')
    results['eda_stats'] = eda_stats
    
    # Salvar histograma de idades
    fig_age = plot_age_histogram(df)
    fig_age.savefig(f'{output_dir}/histograma_idades.png', dpi=150, bbox_inches='tight')
    plt.close(fig_age)
    
    # =========================================================================
    # 2.1 - SARCOMAS POR FAIXA ETÁRIA
    # =========================================================================
    sarcoma_table, fig_sarcoma = analyze_sarcomas_by_age(df)
    fig_sarcoma.savefig(f'{output_dir}/sarcomas_por_idade.png', dpi=150, bbox_inches='tight')
    plt.close(fig_sarcoma)
    sarcoma_table.to_csv(f'{output_dir}/sarcomas_por_idade.csv')
    results['sarcoma_by_age'] = sarcoma_table
    
    # =========================================================================
    # 2.2 - NEOPLASMAS POR FAIXA ETÁRIA
    # =========================================================================
    neoplasm_table, fig_neoplasm = analyze_all_neoplasms_by_age(df)
    fig_neoplasm.savefig(f'{output_dir}/neoplasmas_por_idade.png', dpi=150, bbox_inches='tight')
    plt.close(fig_neoplasm)
    neoplasm_table.to_csv(f'{output_dir}/neoplasmas_por_idade.csv')
    results['neoplasms_by_age'] = neoplasm_table
    
    # =========================================================================
    # 2.3 - TODAS AS CAUSAS POR FAIXA ETÁRIA
    # =========================================================================
    all_causes_table, fig_all = analyze_all_causes_by_age(df)
    fig_all.savefig(f'{output_dir}/todas_causas_por_idade.png', dpi=150, bbox_inches='tight')
    plt.close(fig_all)
    all_causes_table.to_csv(f'{output_dir}/todas_causas_por_idade.csv')
    results['all_causes_by_age'] = all_causes_table
    
    # =========================================================================
    # 2.4 - PERCENTUAL SARCOMAS ÓSSEOS
    # =========================================================================
    bone_pct_table, fig_bone_pct = analyze_bone_sarcoma_percentage(df, all_causes_table)
    fig_bone_pct.savefig(f'{output_dir}/percentual_sarcomas_osseos.png', dpi=150, bbox_inches='tight')
    plt.close(fig_bone_pct)
    bone_pct_table.to_csv(f'{output_dir}/percentual_sarcomas_osseos.csv', index=False)
    results['bone_sarcoma_pct'] = bone_pct_table
    
    # =========================================================================
    # 2.5 - PERCENTUAL SARCOMAS TECIDOS MOLES
    # =========================================================================
    soft_pct_table, fig_soft_pct = analyze_soft_tissue_sarcoma_percentage(df, all_causes_table)
    fig_soft_pct.savefig(f'{output_dir}/percentual_sarcomas_tecidos_moles.png', dpi=150, bbox_inches='tight')
    plt.close(fig_soft_pct)
    soft_pct_table.to_csv(f'{output_dir}/percentual_sarcomas_tecidos_moles.csv', index=False)
    results['soft_tissue_pct'] = soft_pct_table
    
    # =========================================================================
    # 4.1 - SARCOMAS POR RAÇA
    # =========================================================================
    sarcoma_race_table, fig_sarcoma_race = analyze_sarcomas_by_race(df)
    fig_sarcoma_race.savefig(f'{output_dir}/sarcomas_por_raca.png', dpi=150, bbox_inches='tight')
    plt.close(fig_sarcoma_race)
    sarcoma_race_table.to_csv(f'{output_dir}/sarcomas_por_raca.csv')
    results['sarcoma_by_race'] = sarcoma_race_table
    
    # =========================================================================
    # 4.2 - CÂNCER TOTAL POR RAÇA
    # =========================================================================
    cancer_race_table, fig_cancer_race = analyze_all_cancer_by_race(df)
    fig_cancer_race.savefig(f'{output_dir}/cancer_total_por_raca.png', dpi=150, bbox_inches='tight')
    plt.close(fig_cancer_race)
    cancer_race_table.to_csv(f'{output_dir}/cancer_total_por_raca.csv', index=False)
    results['cancer_by_race'] = cancer_race_table
    
    # =========================================================================
    # 4.3a - SARCOMAS ÓSSEOS POR RAÇA
    # =========================================================================
    bone_race_table, fig_bone_race = analyze_bone_sarcomas_by_race(df)
    fig_bone_race.savefig(f'{output_dir}/sarcomas_osseos_por_raca.png', dpi=150, bbox_inches='tight')
    plt.close(fig_bone_race)
    bone_race_table.to_csv(f'{output_dir}/sarcomas_osseos_por_raca.csv')
    results['bone_sarcoma_by_race'] = bone_race_table
    
    # =========================================================================
    # 4.3b - SARCOMAS TECIDOS MOLES POR RAÇA
    # =========================================================================
    soft_race_table, fig_soft_race = analyze_soft_tissue_sarcomas_by_race(df)
    fig_soft_race.savefig(f'{output_dir}/sarcomas_tecidos_moles_por_raca.png', dpi=150, bbox_inches='tight')
    plt.close(fig_soft_race)
    soft_race_table.to_csv(f'{output_dir}/sarcomas_tecidos_moles_por_raca.csv', index=False)
    results['soft_tissue_by_race'] = soft_race_table
    
    # =========================================================================
    # 5.1 - CIDs ESPECÍFICOS EM JOVENS
    # =========================================================================
    young_table, fig_young = analyze_specific_cids_young(df)
    fig_young.savefig(f'{output_dir}/cids_jovens.png', dpi=150, bbox_inches='tight')
    plt.close(fig_young)
    young_table.to_csv(f'{output_dir}/cids_jovens.csv')
    results['young_cids'] = young_table
    
    # =========================================================================
    # 5.4 - ASMR COMPARATIVO
    # =========================================================================
    asmr_table, fig_asmr = analyze_asmr_comparative(df, population_data)
    fig_asmr.savefig(f'{output_dir}/asmr_comparativo.png', dpi=150, bbox_inches='tight')
    plt.close(fig_asmr)
    asmr_table.to_csv(f'{output_dir}/asmr_comparativo.csv', index=False)
    results['asmr'] = asmr_table
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    print("\n" + "="*80)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*80)
    print(f"\n📁 Outputs salvos em: {os.path.abspath(output_dir)}/")
    print(f"📊 Total de linhas processadas: {len(df):,}")
    print(f"📈 Gráficos gerados: 12")
    print(f"📋 Tabelas exportadas: 12")
    
    return results


# ============================================================================
# EXEMPLO DE USO E FUNÇÃO MAIN
# ============================================================================

if __name__ == "__main__":
    # Definir caminho do arquivo
    DATA_FILE = "DataSet-Ano2000a2019_CID-Todos_20251130.csv"
    OUTPUT_DIR = "output_analise"
    
    # Verificar se arquivo existe
    if os.path.exists(DATA_FILE):
        print(f"✅ Arquivo encontrado: {DATA_FILE}")
        
        # Executar análise completa
        # Para teste rápido, use sample_fraction=0.01 (1% dos dados)
        # Para análise completa, use sample_fraction=None
        results = run_full_analysis(
            filepath=DATA_FILE,
            sample_fraction=None,  # Mudar para 0.01 para teste rápido
            output_dir=OUTPUT_DIR,
            population_data=None  # Inserir dados de população se disponíveis
        )
        
        print("\n🎉 Análise finalizada! Verifique a pasta '{}' para resultados.".format(OUTPUT_DIR))
        
    else:
        print(f"⚠️  Arquivo não encontrado: {DATA_FILE}")
        print("\n📝 Instruções:")
        print("1. Coloque o arquivo CSV na mesma pasta deste script")
        print("2. Ou atualize a variável DATA_FILE com o caminho correto")
        print("3. Execute novamente: python analise_obitos.py")
        
        # Criar exemplo com dados sintéticos para demonstração
        print("\n🔬 Criando exemplo com dados sintéticos para demonstração...")
        
        np.random.seed(42)
        n_samples = 10000
        
        synthetic_data = {
            'ano': np.random.choice(range(2000, 2020), n_samples),
            'sigla_uf': np.random.choice(['SP', 'RJ', 'MG', 'RS', 'BA'], n_samples),
            'causa_basica': np.random.choice(['C400', 'C401', 'C410', 'C419', 'C490', 'C499', 
                                              'C500', 'C61', 'I10', 'J18'], n_samples),
            'idade': np.random.exponential(50, n_samples).clip(0, 100),
            'sexo': np.random.choice(['Masculino', 'Feminino'], n_samples),
            'raca_cor': np.random.choice(['Branca', 'Parda', 'Preta', 'Amarela', 'Indígena'], n_samples)
        }
        
        df_synthetic = pd.DataFrame(synthetic_data)
        df_synthetic.to_csv(DATA_FILE, index=False)
        print(f"✅ Dataset sintético criado: {DATA_FILE} ({n_samples} linhas)")
        
        # Executar análise com dados sintéticos
        results = run_full_analysis(
            filepath=DATA_FILE,
            sample_fraction=None,
            output_dir=OUTPUT_DIR
        )
