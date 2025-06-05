import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')

# Paleta de Colores Oceánica Cohesiva - Integrada con el diseño de la app
WAVE_COLORS = [
    '#000033',  # Azul marino profundo (base app)
    '#001f5c',  # Azul océano profundo 
    '#0041a3',  # Azul brillante
    '#0074D9',  # Azul medio (color principal app)
    '#39CCCC',  # Cian/Teal (color accent app)
    '#00e6e6',  # Cian brillante
    '#00ffcc',  # Cian-verde
    '#00ff99',  # Verde-cian
    '#66ff66',  # Verde claro
    '#99ff33',  # Verde-amarillo
    '#ccff00',  # Amarillo-verde
    '#ffff00',  # Amarillo puro
    '#ffcc00',  # Amarillo-naranja
    '#ff9900',  # Naranja
    '#ff6600',  # Naranja-rojo
    '#ff3300'   # Rojo (peak)
]

# Paleta complementaria para elementos UI
UI_COLORS = {
    'primary': '#39CCCC',
    'secondary': '#0074D9', 
    'accent': '#00e6e6',
    'background': '#000033',
    'surface': '#001f5c',
    'text': '#ffffff',
    'text_secondary': '#b0e0e6'
}

# Configuración de página Streamlit
st.set_page_config(
    page_title="🌊 Generador de Superficies de Volatilidad - BQuant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado con tema oceánico cohesivo
st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(90deg, {UI_COLORS['background']}, {UI_COLORS['surface']}, {UI_COLORS['secondary']}, {UI_COLORS['primary']});
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(57, 204, 204, 0.3);
        border: 1px solid {UI_COLORS['accent']};
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, {UI_COLORS['background']}, {UI_COLORS['surface']});
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid {UI_COLORS['primary']};
        box-shadow: 0 3px 15px rgba(0, 116, 217, 0.2);
        border: 1px solid rgba(57, 204, 204, 0.2);
    }}
    
    .stMetric {{
        background: linear-gradient(135deg, rgba(0, 31, 92, 0.15), rgba(57, 204, 204, 0.1));
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid {UI_COLORS['accent']};
        box-shadow: 0 2px 8px rgba(0, 230, 230, 0.1);
    }}
    
    .bquant-footer {{
        background: linear-gradient(90deg, {UI_COLORS['background']}, {UI_COLORS['surface']});
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
        border: 2px solid {UI_COLORS['primary']};
        box-shadow: 0 4px 20px rgba(57, 204, 204, 0.2);
    }}
    
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, {UI_COLORS['background']}, {UI_COLORS['surface']});
    }}
    
    .stButton > button {{
        background: linear-gradient(45deg, {UI_COLORS['primary']}, {UI_COLORS['secondary']});
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 3px 10px rgba(57, 204, 204, 0.3);
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(45deg, {UI_COLORS['accent']}, {UI_COLORS['primary']});
        box-shadow: 0 5px 15px rgba(57, 204, 204, 0.5);
        transform: translateY(-2px);
    }}
    
    .stSelectbox > div > div {{
        background: linear-gradient(135deg, rgba(0, 31, 92, 0.3), rgba(57, 204, 204, 0.1));
        border: 1px solid {UI_COLORS['primary']};
        border-radius: 8px;
    }}
    
    .stTextInput > div > div > input {{
        background: linear-gradient(135deg, rgba(0, 31, 92, 0.2), rgba(57, 204, 204, 0.05));
        border: 1px solid {UI_COLORS['accent']};
        border-radius: 8px;
        color: {UI_COLORS['text']};
    }}
    
    .stSlider > div > div > div {{
        background: {UI_COLORS['primary']};
    }}
    
    /* Información contextual mejorada */
    .info-box {{
        background: linear-gradient(135deg, rgba(0, 230, 230, 0.1), rgba(0, 116, 217, 0.1));
        border-left: 4px solid {UI_COLORS['accent']};
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}
    
    /* Títulos de secciones */
    .section-title {{
        color: {UI_COLORS['primary']};
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {UI_COLORS['accent']};
    }}
</style>
""", unsafe_allow_html=True)

def black_scholes_call(S, K, T, r, sigma):
    """Calcular precio de opción call Black-Scholes con manejo de errores"""
    from scipy.stats import norm
    import math
    
    if T <= 0:
        return max(S - K, 0)
    if sigma <= 0:
        return np.nan
    
    try:
        d1 = (math.log(S/K) + (r + sigma**2/2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return call_price
    except:
        return np.nan

def black_scholes_put(S, K, T, r, sigma):
    """Calcular precio de opción put Black-Scholes con manejo de errores"""
    from scipy.stats import norm
    import math
    
    if T <= 0:
        return max(K - S, 0)
    if sigma <= 0:
        return np.nan
    
    try:
        d1 = (math.log(S/K) + (r + sigma**2/2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    except:
        return np.nan

def implied_volatility_stable(market_price, S, K, T, r, option_type='call'):
    """Calcular volatilidad implícita con enfoque en estabilidad"""
    if T <= 0 or market_price <= 0:
        return np.nan
    
    # Verificaciones de valor intrínseco
    if option_type == 'call':
        intrinsic = max(S - K, 0)
        if market_price < intrinsic * 0.95:
            return np.nan
    else:
        intrinsic = max(K - S, 0)
        if market_price < intrinsic * 0.95:
            return np.nan
    
    def objective(sigma):
        try:
            if option_type == 'call':
                bs_price = black_scholes_call(S, K, T, r, sigma)
            else:
                bs_price = black_scholes_put(S, K, T, r, sigma)
            
            if np.isnan(bs_price):
                return 1e6
            return abs(bs_price - market_price)
        except:
            return 1e6
    
    try:
        result = minimize_scalar(objective, bounds=(0.01, 1.5), method='bounded')
        
        if result.success:
            error = objective(result.x)
            if error < market_price * 0.15:
                return result.x
        
        return np.nan
    except:
        return np.nan

@st.cache_data(ttl=300)  # Cache por 5 minutos
def collect_volatility_data(ticker, risk_free_rate=0.05, quality_level="equilibrado"):
    """Recolectar datos de opciones para creación de superficie de volatilidad"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📊 Obteniendo datos de la acción...")
        progress_bar.progress(0.1)
        
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period="5d")
        
        if hist_data.empty:
            st.error(f"❌ No se pudo obtener datos históricos para {ticker}")
            return None, None
            
        stock_price = hist_data['Close'].iloc[-1]
        
        status_text.text("📅 Obteniendo fechas de vencimiento...")
        progress_bar.progress(0.2)
        
        try:
            all_expirations = list(stock.options)
        except Exception as e:
            st.error(f"❌ No se encontraron opciones para {ticker}. Puede que no tenga opciones listadas o el mercado esté cerrado.")
            return None, None
            
        if not all_expirations:
            st.error(f"❌ No hay fechas de vencimiento disponibles para {ticker}")
            return None, None
        current_date = pd.Timestamp.now()
        
        # Filtrar vencimientos basado en nivel de calidad
        if quality_level == "alta":
            min_days, max_days = 14, 180
            max_expirations = 8
        elif quality_level == "equilibrado":
            min_days, max_days = 7, 365
            max_expirations = 12
        else:  # relajado
            min_days, max_days = 3, 500
            max_expirations = 20
        
        filtered_expirations = []
        for exp_date in all_expirations:
            exp_datetime = pd.to_datetime(exp_date)
            days_to_expiry = (exp_datetime - current_date).days
            if min_days <= days_to_expiry <= max_days:
                filtered_expirations.append(exp_date)
        
        selected_expirations = filtered_expirations[:max_expirations]
        
        if not selected_expirations:
            st.warning(f"⚠️ No se encontraron vencimientos válidos para {ticker} en el rango de {min_days}-{max_days} días")
            return None, None
            
        st.info(f"✅ Encontrados {len(selected_expirations)} vencimientos válidos para analizar")
        
        volatility_data = []
        total_expirations = len(selected_expirations)
        
        for i, exp_date in enumerate(selected_expirations):
            status_text.text(f"🔄 Procesando {exp_date} ({i+1}/{total_expirations})")
            progress_bar.progress(0.2 + (0.7 * i / total_expirations))
            
            try:
                opt_chain = stock.option_chain(exp_date)
                exp_datetime = pd.to_datetime(exp_date)
                days_to_expiry = (exp_datetime - pd.Timestamp.now()).days
                time_to_expiry = days_to_expiry / 365.25
                
                for option_type, options_df in [('call', opt_chain.calls), ('put', opt_chain.puts)]:
                    # Filtrado de calidad basado en nivel
                    if quality_level == "alta":
                        quality_options = options_df[
                            (options_df['volume'].fillna(0) >= 10) &
                            (options_df['bid'] > 0.10) &
                            (options_df['ask'] > 0.10) &
                            ((options_df['ask'] - options_df['bid']) / options_df['ask'] < 0.25) &
                            (options_df['openInterest'].fillna(0) >= 100)
                        ].copy()
                    elif quality_level == "equilibrado":
                        quality_options = options_df[
                            (options_df['volume'].fillna(0) >= 2) &
                            (options_df['bid'] > 0.05) &
                            (options_df['ask'] > 0.05) &
                            ((options_df['ask'] - options_df['bid']) / options_df['ask'] < 0.5)
                        ].copy()
                    else:  # relajado
                        quality_options = options_df[
                            (options_df['bid'] > 0.01) &
                            (options_df['ask'] > 0.01) &
                            (options_df['lastPrice'] > 0)
                        ].copy()
                    
                    for _, row in quality_options.iterrows():
                        strike = row['strike']
                        moneyness = strike / stock_price
                        
                        if not (0.6 <= moneyness <= 1.5):
                            continue
                        
                        market_price = (row['bid'] + row['ask']) / 2
                        iv = implied_volatility_stable(market_price, stock_price, strike,
                                                     time_to_expiry, risk_free_rate, option_type)
                        
                        if not np.isnan(iv) and 0.03 <= iv <= 1.2:
                            volatility_data.append({
                                'strike': strike,
                                'expiry_years': time_to_expiry,
                                'expiry_date': exp_date,
                                'days_to_expiry': days_to_expiry,
                                'moneyness': moneyness,
                                'implied_vol': iv,
                                'option_type': option_type,
                                'market_price': market_price,
                                'volume': row['volume'],
                                'open_interest': row['openInterest']
                            })
            
            except Exception as e:
                st.warning(f"⚠️ Error procesando {exp_date}: {str(e)}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text("✅ Recolección de datos completada!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        df = pd.DataFrame(volatility_data)
        
        if len(df) == 0:
            st.warning(f"⚠️ No se recolectaron datos válidos para {ticker}")
            st.info("""
            **Posibles soluciones:**
            - Intenta durante horas de mercado (9:30 AM - 4:00 PM ET)
            - Usa un ticker más líquido (SPY, QQQ, AAPL)
            - Cambia a nivel de calidad "relajado"
            - Verifica que el ticker tenga opciones listadas
            """)
            return None, None
            
        return df, stock_price
        
    except Exception as e:
        st.error(f"❌ Error recolectando datos: {str(e)}")
        return None, None

def create_plotly_surface(df, ticker, stock_price):
    """Crear superficie de volatilidad 3D interactiva usando Plotly"""
    
    if len(df) == 0:
        return None
    
    # Preparar datos
    moneyness = df['moneyness'].values
    expiry = df['expiry_years'].values
    iv = df['implied_vol'].values
    
    # Crear malla para interpolación
    moneyness_range = np.linspace(moneyness.min(), moneyness.max(), 40)
    expiry_range = np.linspace(expiry.min(), expiry.max(), 40)
    X, Y = np.meshgrid(expiry_range, moneyness_range)
    
    try:
        # Interpolar superficie
        Z = griddata((expiry, moneyness), iv, (X, Y), method='cubic', fill_value=np.nan)
        
        # Rellenar valores NaN
        nan_mask = np.isnan(Z)
        if np.sum(nan_mask) > 0:
            Z_linear = griddata((expiry, moneyness), iv, (X, Y), method='linear', fill_value=np.nan)
            Z = np.where(nan_mask, Z_linear, Z)
        
        # Aplicar suavizado
        Z_smooth = gaussian_filter(Z, sigma=0.8, mode='nearest')
        
        # Crear superficie Plotly con colores mejorados
        surface = go.Surface(
            x=X, y=Y, z=Z_smooth,
            colorscale=[[i/(len(WAVE_COLORS)-1), color] for i, color in enumerate(WAVE_COLORS)],
            opacity=0.95,
            name="Superficie de Volatilidad",
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Volatilidad Implícita",
                    font=dict(color=UI_COLORS['text'])
                ),
                tickfont=dict(color=UI_COLORS['text_secondary']),
                bgcolor=UI_COLORS['surface'],
                bordercolor=UI_COLORS['primary'],
                borderwidth=1
            )
        )
        
        # Añadir puntos de datos
        scatter = go.Scatter3d(
            x=expiry, y=moneyness, z=iv,
            mode='markers',
            marker=dict(
                size=5,
                color=iv,
                colorscale=[[i/(len(WAVE_COLORS)-1), color] for i, color in enumerate(WAVE_COLORS)],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name="Puntos de Datos",
            text=[f"Strike: {s:.2f}<br>IV: {v:.1%}<br>Días: {d}" 
                  for s, v, d in zip(df['strike'], df['implied_vol'], df['days_to_expiry'])],
            hovertemplate="<b>%{text}</b><extra></extra>"
        )
        
        fig = go.Figure(data=[surface, scatter])
        
        fig.update_layout(
            title=dict(
                text=f'{ticker} Superficie de Volatilidad | Precio: ${stock_price:.2f} | Puntos: {len(df):,}',
                font=dict(color=UI_COLORS['text'], size=18),
                x=0.5
            ),
            scene=dict(
                xaxis_title='Tiempo al Vencimiento (Años)',
                yaxis_title='Moneyness (Strike/Spot)',
                zaxis_title='Volatilidad Implícita',
                bgcolor=UI_COLORS['background'],
                xaxis=dict(
                    backgroundcolor=f"rgba({int(UI_COLORS['surface'][1:3], 16)},{int(UI_COLORS['surface'][3:5], 16)},{int(UI_COLORS['surface'][5:7], 16)},0.8)", 
                    gridcolor=UI_COLORS['accent'],
                    title=dict(
                        text='Tiempo al Vencimiento (Años)',
                        font=dict(color=UI_COLORS['text'])
                    ),
                    tickfont=dict(color=UI_COLORS['text_secondary'])
                ),
                yaxis=dict(
                    backgroundcolor=f"rgba({int(UI_COLORS['surface'][1:3], 16)},{int(UI_COLORS['surface'][3:5], 16)},{int(UI_COLORS['surface'][5:7], 16)},0.8)", 
                    gridcolor=UI_COLORS['accent'],
                    title=dict(
                        text='Moneyness (Strike/Spot)',
                        font=dict(color=UI_COLORS['text'])
                    ),
                    tickfont=dict(color=UI_COLORS['text_secondary'])
                ),
                zaxis=dict(
                    backgroundcolor=f"rgba({int(UI_COLORS['surface'][1:3], 16)},{int(UI_COLORS['surface'][3:5], 16)},{int(UI_COLORS['surface'][5:7], 16)},0.8)", 
                    gridcolor=UI_COLORS['accent'],
                    title=dict(
                        text='Volatilidad Implícita',
                        font=dict(color=UI_COLORS['text'])
                    ),
                    tickfont=dict(color=UI_COLORS['text_secondary'])
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            font=dict(color=UI_COLORS['text']),
            paper_bgcolor=UI_COLORS['background'],
            plot_bgcolor=UI_COLORS['background'],
            height=700
        )
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creando superficie: {str(e)}")
        return None

def main():
    # Encabezado mejorado con cohesión visual
    st.markdown(f"""
    <div class="main-header">
        <h1 style="color: {UI_COLORS['text']}; margin: 0; text-align: center; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🌊 Generador de Superficies de Volatilidad</h1>
        <p style="color: {UI_COLORS['primary']}; margin: 0; text-align: center; font-size: 18px; font-weight: 500;">Análisis Interactivo 3D de Volatilidad de Opciones</p>
        <p style="color: {UI_COLORS['accent']}; margin: 0; text-align: center; font-size: 14px; margin-top: 10px; font-weight: 600;">Desarrollado por <strong>BQuant</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Controles de la barra lateral con estilo cohesivo
    st.sidebar.markdown(f'<div class="section-title">🎛️ Controles de Superficie</div>', unsafe_allow_html=True)
    
    # Selección de ticker
    default_tickers = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN"]
    
    ticker_input = st.sidebar.selectbox(
        "📈 Seleccionar Ticker",
        options=default_tickers,
        index=0,
        help="Elige un ticker líquido para mejores resultados"
    )
    
    # Opción de ticker personalizado
    custom_ticker = st.sidebar.text_input(
        "O ingresa ticker personalizado:",
        placeholder="ej. NFLX",
        help="Ingresa cualquier símbolo de ticker"
    )
    
    ticker = custom_ticker.upper() if custom_ticker else ticker_input
    
    # Parámetros con estilo mejorado
    st.sidebar.markdown(f'<div class="section-title">🔧 Parámetros</div>', unsafe_allow_html=True)
    
    risk_free_rate = st.sidebar.slider(
        "Tasa Libre de Riesgo (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="Tasa de interés libre de riesgo actual"
    ) / 100
    
    quality_level = st.sidebar.selectbox(
        "Nivel de Calidad de Datos",
        options=["alta", "equilibrado", "relajado"],
        index=1,
        help="Mayor calidad = menos puntos pero más confiables"
    )
    
    # Opciones avanzadas
    with st.sidebar.expander("🔬 Opciones Avanzadas"):
        min_volume = st.number_input("Volumen Mínimo", value=1, min_value=0)
        max_spread = st.slider("Máx Spread Bid-Ask (%)", 10, 100, 50) / 100
        min_oi = st.number_input("Interés Abierto Mínimo", value=0, min_value=0)
    
    # Botón generar
    generate_button = st.sidebar.button(
        "🚀 Generar Superficie",
        type="primary",
        use_container_width=True
    )
    
    # Información de estado del mercado (simplificado)
    current_hour = datetime.now().hour
    if 9 <= current_hour <= 16:
        st.sidebar.info("💡 Tip: Mejores datos durante horas de mercado (9:30-16:00 ET)")
    else:
        st.sidebar.warning("⏰ Fuera de horario de mercado - datos pueden estar limitados")
    
    # Área de contenido principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f'<div class="section-title">📊 Superficie de Volatilidad</div>', unsafe_allow_html=True)
        surface_container = st.container()
    
    with col2:
        st.markdown(f'<div class="section-title">📈 Datos de Mercado</div>', unsafe_allow_html=True)
        metrics_container = st.container()
        
        st.markdown(f'<div class="section-title">📋 Resumen de Datos</div>', unsafe_allow_html=True)
        summary_container = st.container()
    
    # Generar superficie cuando se presiona el botón
    if generate_button:
        with st.spinner(f"🔄 Generando superficie de volatilidad para {ticker}..."):
            df, stock_price = collect_volatility_data(ticker, risk_free_rate, quality_level)
            
            if df is not None and len(df) > 0:
                # Crear superficie
                fig = create_plotly_surface(df, ticker, stock_price)
                
                if fig is not None:
                    with surface_container:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar métricas
                    with metrics_container:
                        col_m1, col_m2 = st.columns(2)
                        
                        with col_m1:
                            st.metric("💰 Precio Acción", f"${stock_price:.2f}")
                            st.metric("📊 Puntos de Datos", f"{len(df):,}")
                        
                        with col_m2:
                            st.metric("📅 Vencimientos", f"{df['expiry_date'].nunique()}")
                            st.metric("🎯 Rango IV", f"{df['implied_vol'].min():.1%} - {df['implied_vol'].max():.1%}")
                    
                    # Tabla resumen
                    with summary_container:
                        st.markdown("#### 📋 Estadísticas de Superficie")
                        
                        summary_stats = {
                            "Métrica": [
                                "Rango Moneyness",
                                "Rango Temporal (días)",
                                "IV Promedio",
                                "Desviación Estándar IV",
                                "Calls vs Puts"
                            ],
                            "Valor": [
                                f"{df['moneyness'].min():.2f} - {df['moneyness'].max():.2f}",
                                f"{df['days_to_expiry'].min()} - {df['days_to_expiry'].max()}",
                                f"{df['implied_vol'].mean():.1%}",
                                f"{df['implied_vol'].std():.1%}",
                                f"{len(df[df['option_type']=='call'])} / {len(df[df['option_type']=='put'])}"
                            ]
                        }
                        
                        st.dataframe(
                            pd.DataFrame(summary_stats),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Opción de descarga de datos
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Datos",
                            data=csv,
                            file_name=f"{ticker}_datos_volatilidad.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.error("❌ Falló la creación de superficie de volatilidad")
            
            else:
                st.error(f"❌ No hay datos disponibles para {ticker}.")
                st.info("""
                **💡 Consejos para resolver el problema:**
                
                🕐 **Horario**: Intenta durante horas de mercado (9:30 AM - 4:00 PM ET)
                
                📈 **Tickers recomendados**: SPY, QQQ, AAPL, MSFT, TSLA
                
                ⚙️ **Configuración**: Cambia a nivel de calidad "relajado"
                
                🔍 **Verificar**: Asegúrate que el ticker tenga opciones listadas
                """)
                
                # Sugerir tickers alternativos
                suggested_tickers = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]
                if ticker not in suggested_tickers:
                    st.info(f"🔄 **Sugerencia**: Prueba con tickers más líquidos: {', '.join(suggested_tickers)}")
    
    # Sección de información
    st.markdown("---")
    
    with st.expander("ℹ️ Cómo Usar Esta Herramienta"):
        st.markdown("""
        ### 🌊 Guía del Generador de Superficies de Volatilidad
        
        **1. Selecciona tu Ticker** 📈
        - Elige entre tickers populares líquidos (SPY, QQQ, etc.)
        - O ingresa cualquier símbolo de ticker personalizado
        
        **2. Ajusta Parámetros** ⚙️
        - **Tasa Libre de Riesgo**: Tasa actual del tesoro (afecta cálculos de IV)
        - **Nivel de Calidad**: 
          - Alta: Filtros más estrictos, superficie más suave
          - Equilibrado: Buen balance entre calidad y cobertura
          - Relajado: Más puntos de datos, potencialmente más ruidoso
        
        **3. Generar Superficie** 🚀
        - Haz clic en "Generar Superficie" para obtener datos de opciones en vivo
        - Aparecerá superficie 3D interactiva con puntos de datos
        
        **4. Interpretar Resultados** 🔍
        - **Áreas rojas/púrpuras**: Alta volatilidad implícita (mercado espera grandes movimientos)
        - **Áreas azules**: Baja volatilidad implícita (mercado espera calma)
        - **Forma de superficie**: Muestra patrones de sonrisa/sesgo de volatilidad
        
        **Consejos para Mejores Resultados** 💡
        - Usa tickers líquidos durante horas de mercado
        - SPY típicamente tiene las opciones más líquidas
        - Configuraciones de mayor calidad funcionan mejor para acciones populares
        """)
    
    with st.expander("🔬 Detalles Técnicos"):
        st.markdown("""
        ### Marco Matemático
        
        **Fórmula Black-Scholes** 📐
        - Usada para calcular precios teóricos de opciones
        - Invertida numéricamente para encontrar volatilidad implícita
        
        **Procesamiento de Datos** 🔄
        - Opciones filtradas por volumen, spread e interés abierto
        - Interpolación cúbica para creación de superficie suave
        - Filtrado gaussiano para reducir ruido
        
        **Visualización** 🎨
        - Paleta de colores oceánica para interpretación intuitiva
        - Superficie 3D interactiva con Plotly
        - Datos en tiempo real de Yahoo Finance
        
        **Métricas de Calidad** ✅
        - Solo opciones líquidas usadas para cálculos confiables de IV
        - Manejo de errores para casos extremos y anomalías de mercado
        - Filtrado de moneyness (0.6 a 1.5) para rangos realistas
        
        **Desarrollado por BQuant** 🏢
        - Herramienta profesional de análisis cuantitativo
        - Código optimizado para rendimiento y precisión
        - Interfaz intuitiva para traders e investigadores
        """)
    
    # Footer con atribución BQuant - diseño cohesivo
    st.markdown(f"""
    <div class="bquant-footer">
        <h3 style="color: {UI_COLORS['primary']}; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🌊 BQuant - Análisis Cuantitativo Profesional</h3>
        <p style="color: {UI_COLORS['text']}; margin: 5px 0; font-weight: 400;">Herramientas avanzadas para trading e investigación financiera</p>
        <p style="color: {UI_COLORS['accent']}; margin: 0; font-size: 14px; font-weight: 500;">Superficies de Volatilidad | Black-Scholes | Análisis de Opciones</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
