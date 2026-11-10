import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from prophet import Prophet

from datos import cargar_datos

# Configuración de la página
st.set_page_config(
    page_title="Pronóstico de Pasajeros de Avión",
    page_icon="✈️",
    layout="wide"
)

# Cargar los datos
@st.cache_data
def obtener_datos():
    df = cargar_datos()
    df['Fecha'] = pd.to_datetime(df['Month'])
    df['Año'] = df['Fecha'].dt.year
    df['Mes_num'] = df['Fecha'].dt.month
    df['Nombre_Mes'] = df['Fecha'].dt.strftime('%B')
    return df

try:
    df = obtener_datos()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Título principal
st.title("¿Cuántos pasajeros tendremos el próximo año?")
st.markdown("Una aplicación sencilla para entender el pasado y comparar múltiples modelos de pronóstico de pasajeros de avión.")

# -------------------------------------------------------------
# Indicadores superiores (KPIs)
# -------------------------------------------------------------
pasajeros_1949 = df[df['Año'] == 1949]['Passengers'].sum()
pasajeros_1960 = df[df['Año'] == 1960]['Passengers'].sum()
crecimiento_veces = pasajeros_1960 / pasajeros_1949

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="✈️ Pasajeros en 1949 (Total Año)", value=f"{pasajeros_1949:,.0f}")
with col2:
    st.metric(label="🚀 Pasajeros en 1960 (Total Año)", value=f"{pasajeros_1960:,.0f}")
with col3:
    st.metric(label="📈 Cuántas veces creció", value=f"{crecimiento_veces:.1f}x")

st.markdown("---")

# -------------------------------------------------------------
# Pestañas
# -------------------------------------------------------------
pestana1, pestana2, pestana3, pestana4 = st.tabs([
    "1. La historia", 
    "2. La temporada", 
    "3. El examen", 
    "4. El pronóstico"
])

# -------------------------------------------------------------
# PESTAÑA 1: La historia
# -------------------------------------------------------------
with pestana1:
    st.subheader("Evolución mensual de los pasajeros")
    
    fig_historia = px.line(
        df, 
        x='Fecha', 
        y='Passengers',
        labels={'Fecha': 'Año', 'Passengers': 'Número de Pasajeros'},
        title="Historial de Pasajeros (1949 - 1960)"
    )
    fig_historia.update_traces(line_color='#1f77b4', line_width=2.5)
    st.plotly_chart(fig_historia, use_container_width=True)
    
    st.info(
        "Este gráfico muestra cómo ha crecido el número de pasajeros mes a mes a lo largo de los años. "
        "Se observa una tendencia clara de crecimiento constante combinada con subidas y bajadas repetitivas cada año. "
        "Para el negocio, esto significa que la demanda de vuelos aumenta año con año, requiriendo más aviones y personal."
    )

# -------------------------------------------------------------
# PESTAÑA 2: La temporada
# -------------------------------------------------------------
with pestana2:
    st.subheader("Comportamiento según el mes del año")
    
    promedio_mes = df.groupby(['Mes_num', 'Nombre_Mes'])['Passengers'].mean().reset_index()
    promedio_general = df['Passengers'].mean()
    
    fig_temporada = px.bar(
        promedio_mes,
        x='Nombre_Mes',
        y='Passengers',
        labels={'Nombre_Mes': 'Mes', 'Passengers': 'Promedio de Pasajeros'},
        title="Promedio de Pasajeros por Mes (Temporada Alta y Baja)"
    )
    fig_temporada.add_hline(
        y=promedio_general, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Promedio general ({promedio_general:.1f})", 
        annotation_position="bottom right"
    )
    st.plotly_chart(fig_temporada, use_container_width=True)
    
    st.info(
        "Las barras muestran cuáles meses tienen más o menos pasajeros en promedio durante el año, comparados con la línea roja del promedio general. "
        "Los meses de verano (como julio y agosto) superan claramente el promedio, marcando la temporada alta de viajes. "
        "Para el negocio, esto permite planear promociones en temporada baja y asegurar suficiente flota y personal en temporada alta."
    )

# -------------------------------------------------------------
# PESTAÑA 3: El examen (Múltiples modelos evaluados en los últimos 24 meses)
# -------------------------------------------------------------
with pestana3:
    st.subheader("Evaluación de múltiples modelos con los últimos 24 meses")
    
    datos_entrenamiento = df.iloc[:-24].copy()
    datos_prueba = df.iloc[-24:].copy()
    
    # 1. Copiar el año pasado
    pred_copiar = []
    for i in range(len(datos_prueba)):
        idx_historico = len(df) - 24 + i - 12
        pred_copiar.append(df.iloc[idx_historico]['Passengers'])
    datos_prueba['Pred_Copiar'] = pred_copiar
    
    # 2. Suavizamiento Exponencial (Holt-Winters)
    try:
        hw_model = ExponentialSmoothing(
            datos_entrenamiento['Passengers'],
            trend='add',
            seasonal='mul',
            seasonal_periods=12
        ).fit()
        datos_prueba['Pred_HW'] = hw_model.forecast(24).values
    except Exception:
        datos_prueba['Pred_HW'] = datos_prueba['Pred_Copiar']

    # 3. SARIMA
    try:
        sarima_model = SARIMAX(
            datos_entrenamiento['Passengers'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        datos_prueba['Pred_SARIMA'] = sarima_model.forecast(24).values
    except Exception:
        datos_prueba['Pred_SARIMA'] = datos_prueba['Pred_Copiar']

    # 4. Regresión Lineal
    try:
        X_train = np.arange(len(datos_entrenamiento)).reshape(-1, 1)
        y_train = datos_entrenamiento['Passengers'].values
        lr_model = LinearRegression().fit(X_train, y_train)
        
        X_test = np.arange(len(datos_entrenamiento), len(df)).reshape(-1, 1)
        datos_prueba['Pred_LR'] = lr_model.predict(X_test)
    except Exception:
        datos_prueba['Pred_LR'] = datos_prueba['Pred_Copiar']

    # 5. Prophet
    try:
        df_prophet = datos_entrenamiento[['Fecha', 'Passengers']].rename(columns={'Fecha': 'ds', 'Passengers': 'y'})
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        prophet_model.fit(df_prophet)
        
        future = datos_prueba[['Fecha']].rename(columns={'Fecha': 'ds'})
        forecast = prophet_model.predict(future)
        datos_prueba['Pred_Prophet'] = forecast['yhat'].values
    except Exception:
        datos_prueba['Pred_Prophet'] = datos_prueba['Pred_Copiar']

    # Gráfico comparativo
    fig_examen = go.Figure()
    reciente_ref = df.iloc[-36:-24]
    fig_examen.add_trace(go.Scatter(x=reciente_ref['Fecha'], y=reciente_ref['Passengers'], mode='lines', name='Historial anterior', line=dict(color='gray', dash='dot')))
    fig_examen.add_trace(go.Scatter(x=datos_prueba['Fecha'], y=datos_prueba['Passengers'], mode='lines+markers', name='Real (Oculto)', line=dict(color='black', width=3)))
    
    fig_examen.add_trace(go.Scatter(x=datos_prueba['Fecha'], y=datos_prueba['Pred_Copiar'], mode='lines', name='Copiar el año pasado', line=dict(color='orange', dash='dash')))
    fig_examen.add_trace(go.Scatter(x=datos_prueba['Fecha'], y=datos_prueba['Pred_HW'], mode='lines', name='Suavizamiento Exponencial', line=dict(color='green', width=2)))
    fig_examen.add_trace(go.Scatter(x=datos_prueba['Fecha'], y=datos_prueba['Pred_SARIMA'], mode='lines', name='SARIMA', line=dict(color='purple', width=2)))
    fig_examen.add_trace(go.Scatter(x=datos_prueba['Fecha'], y=datos_prueba['Pred_LR'], mode='lines', name='Regresión Lineal', line=dict(color='red', dash='dot')))
    fig_examen.add_trace(go.Scatter(x=datos_prueba['Fecha'], y=datos_prueba['Pred_Prophet'], mode='lines', name='Prophet', line=dict(color='blue', width=2)))
    
    fig_examen.update_layout(title="Comparativa de Múltiples Modelos vs Realidad (Últimos 24 meses)")
    st.plotly_chart(fig_examen, use_container_width=True)
    
    # Calcular errores (MAPE)
    def calc_mape(real, pred):
        return np.mean(np.abs((real - pred) / real)) * 100

    err_copiar = calc_mape(datos_prueba['Passengers'], datos_prueba['Pred_Copiar'])
    err_hw = calc_mape(datos_prueba['Passengers'], datos_prueba['Pred_HW'])
    err_sarima = calc_mape(datos_prueba['Passengers'], datos_prueba['Pred_SARIMA'])
    err_lr = calc_mape(datos_prueba['Passengers'], datos_prueba['Pred_LR'])
    err_prophet = calc_mape(datos_prueba['Passengers'], datos_prueba['Pred_Prophet'])
    
    tabla_errores = pd.DataFrame({
        "Modelo": [
            "Copiar el año pasado", 
            "Suavizamiento Exponencial", 
            "SARIMA", 
            "Regresión Lineal", 
            "Prophet"
        ],
        "Error promedio (%)": [
            f"{err_copiar:.2f}%", 
            f"{err_hw:.2f}%", 
            f"{err_sarima:.2f}%", 
            f"{err_lr:.2f}%", 
            f"{err_prophet:.2f}%"
        ],
        "Valor Error": [err_copiar, err_hw, err_sarima, err_lr, err_prophet]
    })
    
    # Encontrar ganador
    ganador_idx = tabla_errores['Valor Error'].idxmin()
    nombre_ganador = tabla_errores.loc[ganador_idx, 'Modelo']
    
    st.table(tabla_errores[['Modelo', 'Error promedio (%)']])
    st.success(f"🏆 **El modelo ganador fue: {nombre_ganador}**, ya que obtuvo el menor porcentaje de error prediciendo los últimos 24 meses.")
    
    st.info(
        "Evaluamos cinco métodos diferentes ocultando los últimos dos años para ver cuál se equivoca menos. "
        "La Regresión Lineal solo ve una línea recta y falla en capturar la temporada, mientras que modelos avanzados como SARIMA, Prophet o Suavizamiento Exponencial consideran el crecimiento y los ciclos del año. "
        "Para el negocio, esto te ayuda a elegir con base científica cuál modelo usar para planificar el próximo año."
    )

# -------------------------------------------------------------
# PESTAÑA 4: El pronóstico (Selección de modelo para 1961)
# -------------------------------------------------------------
with pestana4:
    st.subheader("Pronóstico para los 12 meses de 1961")
    
    modelo_elegido = st.selectbox(
        "Selecciona el modelo para generar el pronóstico:",
        ["Suavizamiento Exponencial", "SARIMA", "Prophet", "Regresión Lineal", "Copiar el año pasado"]
    )
    
    fechas_1961 = pd.date_range(start='1961-01-01', periods=12, freq='MS')
    
    if modelo_elegido == "Suavizamiento Exponencial":
        mod = ExponentialSmoothing(df['Passengers'], trend='add', seasonal='mul', seasonal_periods=12).fit()
        pred_vals = mod.forecast(12).values
    elif modelo_elegido == "SARIMA":
        mod = SARIMAX(df['Passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred_vals = mod.forecast(12).values
    elif modelo_elegido == "Prophet":
        df_p = df[['Fecha', 'Passengers']].rename(columns={'Fecha': 'ds', 'Passengers': 'y'})
        mod = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        mod.fit(df_p)
        future = pd.DataFrame({'ds': fechas_1961})
        pred_vals = mod.predict(future)['yhat'].values
    elif modelo_elegido == "Regresión Lineal":
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Passengers'].values
        mod = LinearRegression().fit(X, y)
        X_fut = np.arange(len(df), len(df) + 12).reshape(-1, 1)
        pred_vals = mod.predict(X_fut)
    else:  # Copiar el año pasado
        pred_vals = df.iloc[-12]['Passengers'] # fallback o extraer los últimos 12 meses de 1960
        pred_vals = df.iloc[-12:].reset_index()['Passengers'].values # exactamente los 12 meses de 1960

    df_pronostico = pd.DataFrame({
        'Fecha': fechas_1961,
        'Mes': fechas_1961.strftime('%Y-%m'),
        'Passengers_Esperados': pred_vals
    })
    
    # Gráfico del pronóstico
    fig_pronostico = go.Figure()
    ultimo_anio = df.tail(12)
    fig_pronostico.add_trace(go.Scatter(x=ultimo_anio['Fecha'], y=ultimo_anio['Passengers'], mode='lines+markers', name='Historial 1960', line=dict(color='blue')))
    fig_pronostico.add_trace(go.Scatter(x=df_pronostico['Fecha'], y=df_pronostico['Passengers_Esperados'], mode='lines+markers', name=f'Pronóstico 1961 ({modelo_elegido})', line=dict(color='green', width=3, dash='dash')))
    
    fig_pronostico.update_layout(title=f"Pronóstico de Pasajeros para 1961 usando {modelo_elegido}")
    st.plotly_chart(fig_pronostico, use_container_width=True)
    
    tabla_mostrar = df_pronostico[['Mes', 'Passengers_Esperados']].copy()
    tabla_mostrar.columns = ['Mes', 'Pasajeros Esperados']
    tabla_mostrar['Pasajeros Esperados'] = tabla_mostrar['Pasajeros Esperados'].round(0).astype(int)
    
    st.table(tabla_mostrar)
    
    st.info(
        f"Este gráfico y tabla muestran la proyección exacta mes a mes para todo el año 1961 utilizando el modelo **{modelo_elegido}**. "
        "Puedes cambiar de modelo arriba para comparar visualmente cómo proyecta cada técnica el futuro. "
        "Para el negocio, esto te da la flexibilidad de evaluar diferentes escenarios de demanda para presupuestos y operaciones."
    )
