import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ElasticidadSKU import ElasticidadCB

st.set_page_config(page_title="Elasticidades SKU", layout="wide")

st.title("Elasticidades por SKU")
st.markdown("Sube un layout o captura manualmente los SKUs para calcular elasticidades.")

# Selección de método de carga
opcion = st.radio("Selecciona cómo quieres cargar los SKUs:", 
                  ["Subir Layout", "Capturar Manualmente"])

layout = None

if opcion == "Subir Layout":
    archivo = st.file_uploader("Sube un archivo CSV o Excel con columnas: SKU, Canal, Clima", 
                               type=["csv", "xlsx"])
    if archivo:
        if archivo.name.endswith(".csv"):
            layout = pd.read_csv(archivo)
        else:
            layout = pd.read_excel(archivo)
        st.success("Layout cargado correctamente")
        st.dataframe(layout)
elif opcion == "Capturar Manualmente":
    st.markdown("Agrega un SKU, selecciona canal y clima:")

    # Entradas
    sku = st.text_input("SKU (código de barras)")
    canal = st.selectbox("Canal", ["Moderno", "Autoservicios", "Farmacias"])
    clima = st.checkbox("¿Considerar Clima?", value=True)

    # Inicializar lista en session_state si no existe
    if "manual_layout" not in st.session_state:
        st.session_state.manual_layout = []

    # Botón para agregar SKU a la lista temporal
    if st.button("Agregar SKU a la lista"):
        if sku:  # Solo si hay valor
            st.session_state.manual_layout.append({
                "SKU": sku,
                "Canal": canal,
                "Clima": clima
            })
            st.success(f"SKU {sku} agregado a la lista.")
        else:
            st.warning("Debes ingresar un SKU válido.")

    # Mostrar la tabla de SKUs capturados hasta el momento
    if st.session_state.manual_layout:
        st.markdown("### SKUs capturados:")
        layout = pd.DataFrame(st.session_state.manual_layout)
        st.dataframe(layout)


# Procesar layout
if layout is not None and st.button("Ejecutar Análisis"):
    resultados = []
    graficos = {}

    with st.spinner("Calculando elasticidades"):
        for _, row in layout.iterrows():
            sku = row["SKU"]
            canal = row["Canal"]
            temp = row["Clima"]

            try:
                elasticidad = ElasticidadCB(codbarras=sku, canal=canal, temp=temp)
                elasticidad.consulta_sellout()
                elasticidad.calcula_elasticidad()
                fig = elasticidad.grafica()
                graficos[sku] = fig
                insight = elasticidad.genera_insight()
                def safe_round(value, dec=4):
                    return round(value, dec) if value is not None else None

                resultados.append({
                    'SKU': sku,
                    'Canal': canal,
                    'Intercepto': safe_round(elasticidad.coeficientes.get('Intercept'), 2),
                    'Coef. Precio': safe_round(elasticidad.coeficientes.get('Precio'), 4),
                    'Coef. Clima': safe_round(elasticidad.coeficientes.get('CLIMA'), 4),
                    'Pvalue Intercepto': safe_round(elasticidad.pvalores.get('Intercept'), 4),
                    'Pvalue Precio': safe_round(elasticidad.pvalores.get('Precio'), 4),
                    'Pvalue Clima': safe_round(elasticidad.pvalores.get('CLIMA'), 4),
                    'R cuadrada': safe_round(elasticidad.r2, 3),
                    "Insight": insight
                })



            except Exception as e:
                st.error(f"Error en SKU {sku}: {e}")

    if resultados:
        df_resultados = pd.DataFrame(resultados)

        

        st.subheader(" Gráficos e Insights por SKU")
        for res in resultados:
            sku = res["SKU"]
            
            with st.expander(f" SKU {sku} - Canal {res['Canal']}"):
                
                col1, col2 = st.columns([2, 1]) 
                
                with col1:
                    st.markdown("## Resumen")
                    df_sku =df_resultados[['Intercepto','Coef. Precio','Coef. Clima','Pvalue Intercepto','Pvalue Precio','Pvalue Clima','R cuadrada']][df_resultados['SKU']==sku]
                    st.dataframe(df_sku)
                    st.markdown("")
                    st.markdown("## Gráfico")
                    if sku in graficos:
                        st.pyplot(graficos[sku]) 

                with col2:
                    st.markdown("## Insight")
                    st.info(res["Insight"])

               
