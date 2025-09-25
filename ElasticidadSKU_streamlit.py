import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
from ElasticidadSKU import ElasticidadCB
import numpy as np

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
    # PRODUCTOS 
    conn = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"]
        )

    query = f"""SELECT MRCNOMBRE AS MARCA,
                    AGPPAUTANOMBRE AS AGRUPACION_PAUTA,
                    PRONOMBRE AS PRODUCTO_BASE,
                    PROPSTCODBARRAS AS SKU, 
                    PROPSTNOMBRE AS PRODUCTO
                FROM PRD_CNS_MX.DM.VW_DIM_PRODUCTO"""
    df_productos =  pd.read_sql(query,conn)
    conn.close()
    st.markdown("Agrega un SKU, selecciona canal y clima:")
    # --- FILTROS EN UNA FILA ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        marca = st.selectbox("Marca", sorted(df_productos["MARCA"].unique()))

    with col2:
        agrupaciones = df_productos[df_productos["MARCA"] == marca]["AGRUPACION_PAUTA"].unique()
        agrupacion = st.selectbox("Agrupación", sorted(agrupaciones))

    with col3:
        productos_base = df_productos[
            (df_productos["MARCA"] == marca) &
            (df_productos["AGRUPACION_PAUTA"] == agrupacion)
        ]["PRODUCTO_BASE"].unique()
        producto_base = st.selectbox("Producto Base", sorted(productos_base))

    with col4:
        skus_filtrados = df_productos[
            (df_productos["MARCA"] == marca) &
            (df_productos["AGRUPACION_PAUTA"] == agrupacion) &
            (df_productos["PRODUCTO_BASE"] == producto_base)
        ][["SKU", "PRODUCTO"]]

        sku_row = st.selectbox(
            "SKU",
            skus_filtrados.apply(lambda x: f"{x['SKU']} - {x['PRODUCTO']}", axis=1)
        )


    canal = st.selectbox("Canal", ["Moderno", "Autoservicios", "Farmacias"])
    
    precio_act = st.text_input("Si conoces el precio agregalo")

    # Validamos que sea número
    if precio_act.replace(".", "", 1).isdigit():  
        precio = float(precio_act)
    elif precio_act == "":
        precio =""
    else:
        st.error("⚠️ Solo se permiten números (ej: 123 o 123.45)")

    clima = st.checkbox("¿Considerar Clima?", value=True)    

    # Inicializar lista en session_state
    if "manual_layout" not in st.session_state:
        st.session_state.manual_layout = []

    # Botón para agregar
    if st.button("Agregar SKU a la lista"):
        prod = sku_row.split(" - ")[1] 
        sku_val = sku_row.split(" - ")[0]  # extraer el código de barras
        st.session_state.manual_layout.append({
            "SKU": sku_val,
            "PropstNombre":prod,
            "Canal": canal,
            "Clima": clima,
            "Precio Actual": precio
        })
        st.success(f"SKU {sku_val} agregado a la lista.")

    # Mostrar tabla acumulada
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
            prod = row["PropstNombre"]

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
                    'Producto': prod,
                    #intercepto
                    'Venta Base': safe_round(np.exp(elasticidad.coeficientes.get('Intercept')), 0),
                    #coeficientes
                    'Afectación Precio': safe_round(elasticidad.coeficientes.get('Precio'), 4),
                    'Afectación Clima': safe_round(elasticidad.coeficientes.get('CLIMA'), 4),
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
            prod = res["Producto"]
            venta_base = res['Venta Base']
            af_precio = res['Afectación Precio']
            af_clima = res['Afectación Clima']
            r2 = res['R cuadrada']
            
            with st.expander(f" SKU {sku} - {prod} - Canal {res['Canal']}"):
                
                col1, col2 = st.columns([2, 1]) 
                
                with col1:
                    st.markdown("## Resumen")
                    df_sku =df_resultados[['Venta Base','Afectación Precio','Afectación Clima','Pvalue Intercepto','Pvalue Precio','Pvalue Clima','R cuadrada']][df_resultados['SKU']==sku]
                    st.dataframe(df_sku)
                    st.markdown(f"""
                                📦 **Producto:** {prod}  
                                🆔 **SKU:** {sku}  
                                🏬 **Canal:** {canal}  

                                - 📊 **Ventas base esperadas:** {venta_base:,} unidades (sin considerar precio ni clima).  
                                - 💰 **Elasticidad precio:** {af_precio}.  
                                Esto significa que si el precio aumenta 1%, la venta cambia en aproximadamente **{af_precio}**%.  
                                - 🌦️ **Impacto del clima:** {af_clima}.   
                                Por cada 1% de incremento en la temperatura el sellout cambia en un **{af_clima:.2%}**.
                                - 📈 **Calidad del modelo (R²):** {r2}.  
                                El modelo explica un **{r2*100}**% de la variacion de la venta.
                                """)
                    st.markdown("")
                    st.markdown("## Gráfico")
                    if sku in graficos:
                        st.pyplot(graficos[sku]) 

                with col2:
                    st.markdown("## Insight")
                    st.markdown(
                                f"""
                                <div style="
                                    border: .05px solid gray;   /* Borde gris */
                                    padding: 10px;             /* Espacio interno */
                                    border-radius: 8px;        /* Esquinas redondeadas */
                                    background-color: transparent;  /* Fondo transparente */
                                    margin: 10px 0px;          /* Margen arriba y abajo */
                                ">
                                    {res["Insight"]}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
               
