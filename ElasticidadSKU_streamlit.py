import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
from ElasticidadSKU import ElasticidadCB
import numpy as np
import plotly.express as px

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

    #query = f"""SELECT DISTINCT 
    #                MRCNOMBRE AS MARCA,
    #                AGPPAUTANOMBRE AS AGRUPACION_PAUTA,
    #                PRONOMBRE AS PRODUCTO_BASE,
    #                PROPSTCODBARRAS AS SKU, 
    #                PROPSTNOMBRE AS PRODUCTO
    #            FROM PRD_CNS_MX.DM.FACT_DESPLAZAMIENTOSEMANALCADENASKU AS m
    #            LEFT JOIN PRD_CNS_MX.DM.VW_DIM_CLIENTE AS c ON m.CteID = c.CteID
    #            LEFT JOIN PRD_CNS_MX.DM.VW_DIM_PRODUCTO AS p ON m.ProdID = p.ProdID
    #            LEFT JOIN PRD_CNS_MX.DM.VW_DIM_TIEMPO AS t ON m.TMPID = t.TMPID
    #                WHERE t.anio >= 2023"""

    query = f"""  SELECT distinct p.propstcodbarras as SKU,
        p.propstid AS PROPST_ID
        FROM PRD_CNS_MX.DM.FACT_DESPLAZAMIENTOSEMANALCADENASKU AS m
        LEFT JOIN PRD_CNS_MX.DM.VW_DIM_CLIENTE AS c ON m.CteID = c.CteID
        LEFT JOIN PRD_CNS_MX.DM.VW_DIM_PRODUCTO AS p ON m.ProdID = p.ProdID
        LEFT JOIN PRD_CNS_MX.DM.VW_DIM_TIEMPO AS t ON m.TMPID = t.TMPID
        WHERE t.anio >= 2023
          AND c.TIPOESTNOMBRE IN ('Autoservicios','Cadenas de farmacia')
          AND c.TIPOCLIENTE='Monitoreado'"""
    df_propstid =  pd.read_sql(query,conn)
    #conn.close()
    df_productos = pd.read_excel(r'Catálogo Corporativo Final.xlsx',sheet_name='Catálogo')
    df_productos = df_productos[df_productos['IdPais']==1].copy()
    df_productos = df_propstid.merge(df_productos,left_on='PROPST_ID',right_on='ProPstID',how='left')
    st.markdown("Agrega un SKU, selecciona canal y clima:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        marca = st.selectbox("Marca", sorted(df_productos["Marca"].unique()))

    with col2:
        agrupaciones = df_productos[df_productos["Marca"] == marca]["Agrupación Pauta"].unique()
        agrupacion = st.selectbox("Agrupación", sorted(agrupaciones))

    with col3:
        productos_base = df_productos[
            (df_productos["Marca"] == marca) &
            (df_productos["Agrupación Pauta"] == agrupacion)
        ]["Producto Base"].unique()
        producto_base = st.selectbox("Producto Base", sorted(productos_base))

    with col4:
        skus_filtrados = df_productos[
            (df_productos["Marca"] == marca) &
            (df_productos["Agrupación Pauta"] == agrupacion) &
            (df_productos["Producto Base"] == producto_base)
        ][["SKU", "ProPstNombre"]]

        sku_row = st.selectbox(
            "SKU",
            skus_filtrados.apply(lambda x: f"{x['SKU']} - {x['ProPstNombre']}", axis=1)
        )
        
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        canal = st.selectbox("Canal", ["Moderno", "Autoservicios", "Farmacias"])
    with col2:
        precio_act = st.text_input("Precio (opcional)")
    with col3:
        # Cargar los costos desde Excel
        costos = pd.read_excel(r"CostoGestionMensual_2025-10-28-0942 (1).xlsx")
        costos = costos.rename(columns={
            'CODIGOBARRAS': 'PROPSTCODBARRAS',
            'COSTO_GESTION': 'Costo'
        })  
        costos = costos[['PROPSTCODBARRAS', 'Costo']].drop_duplicates()
        sku_val_prov = sku_row.split(" - ")[0] 
        # Filtrar por el SKU actual
        costo_filtrado = costos.loc[costos['PROPSTCODBARRAS'] == sku_val_prov, 'Costo']

        # Si el SKU existe en el archivo, precargar su costo
        if not costo_filtrado.empty:
            costo_default = costo_filtrado.iloc[0]
        else:
            costo_default = ""

        # Mostrar el campo editable con el valor precargado
        costo_act = st.text_input("Costo (opcional)", value=costo_default)
    with col4:  
        st.markdown("Clima")  
        clima = st.checkbox("¿Considerar Clima?", value=True)    


    # AGREGAMOS LA COMPETENCIA SI EXISTE 
    col1,col2 = st.columns(2)
    with col1:
        sku_val_prov = sku_row.split(" - ")[0] 
        # Cargar datos de competencia
        comp = pd.read_excel(r"Competencias_Elasticidades.xlsx")
        comp.columns = [c.strip() for c in comp.columns]
        comp = comp.rename(columns={
            'SKU': 'PROPSTCODBARRAS',
            'Descripcion Competencia': 'DESC_COMPETENCIA',
            'Precio Competencia': 'PRECIO_COMPETENCIA'
        })
        comp = comp[['PROPSTCODBARRAS','ANIO','DESC_COMPETENCIA','SEMNUMERO','PRECIO_COMPETENCIA']]
        comp = comp[comp['PROPSTCODBARRAS'] == sku_val_prov]  # filtrar por SKU

        # Si hay competencia, permitir seleccionar
        if not comp.empty:
            # Mostrar un selectbox con las descripciones únicas de competidores
            descs_comp = comp['DESC_COMPETENCIA'].unique().tolist()
            seleccion_comp = st.selectbox("Selecciona competencia", options=descs_comp)

            # Filtrar la fila seleccionada
            comp_sel = comp[comp['DESC_COMPETENCIA'] == seleccion_comp]

            # Tomar el precio de esa competencia (puedes tomar el último o el promedio)
            precio_comp = comp_sel['PRECIO_COMPETENCIA'].mean()  # o .iloc[-1]

            st.write(f"**Precio competencia seleccionado:** {precio_comp:.2f}")

            # Aquí podrías guardarlo o usarlo en tu tabla principal
            # Ejemplo:
            # df_resultado.loc[sku_row, 'PRECIO_COMPETENCIA'] = precio_comp
        else:
            st.info("No hay información de competencia para este SKU.")


    # Inicializar lista en session_state
    if "manual_layout" not in st.session_state:
        st.session_state.manual_layout = []

    # Botón para agregar
    if st.button("Agregar SKU a la lista"):
        # Validamos que sea número
        if precio_act.replace(".", "", 1).isdigit():  
            precio = float(precio_act)
            # Solo convierte costo si tiene un valor numérico
            if costo_act and costo_act.replace(".", "", 1).isdigit():
                costo = float(costo_act)
            else:
                costo = ""
            prod = sku_row.split(" - ")[1] 
            sku_val = sku_row.split(" - ")[0]  # extraer el código de barras
            st.session_state.manual_layout.append({
                "SKU": sku_val,
                "PropstNombre":prod,
                "Canal": canal,
                "Clima": clima,
                "Precio Actual": precio,
                "Costo Actual": costo
            })
            st.success(f"SKU {sku_val} agregado a la lista.")
        elif precio_act == "":
            precio =""
            costo =""
            prod = sku_row.split(" - ")[1] 
            sku_val = sku_row.split(" - ")[0]  # extraer el código de barras
            st.session_state.manual_layout.append({
                "SKU": sku_val,
                "PropstNombre":prod,
                "Canal": canal,
                "Clima": clima,
                "Precio Actual": precio,
                "Costo Actual": costo
            })
            st.success(f"SKU {sku_val} agregado a la lista.")
        else:
            st.error("⚠️ Solo se permiten números (ej: 123 o 123.45)")
        

    # Mostrar tabla acumulada
    if st.session_state.manual_layout:
        st.markdown("### SKUs capturados:")
        layout = pd.DataFrame(st.session_state.manual_layout)
        st.dataframe(layout)



# Procesar layout
if layout is not None and st.button("Ejecutar Análisis"):
    resultados = []
    graficos = {}
    graficos_dispersion = {}

    with st.spinner("Calculando elasticidades"):
        for _, row in layout.iterrows():
            sku = row["SKU"]
            canal = row["Canal"]
            temp = row["Clima"]
            prod = row["PropstNombre"]
            precioact = row["Precio Actual"]
            costoact = row['Costo Actual']

            try:
                elasticidad = ElasticidadCB(codbarras=sku, canal=canal, temp=temp)
                elasticidad.consulta_sellout()
                elasticidad.calcula_elasticidad()
                fig = elasticidad.grafica()
                dispersion = elasticidad.grafica_dispersion()
                graficos[sku] = fig
                graficos_dispersion[sku] = dispersion
                #insight = elasticidad.genera_insight_op()
                def safe_round(value, dec=4):
                    return round(value, dec) if value is not None else None
                
                resultados.append({
                    'SKU': sku,
                    'Canal': canal,
                    'Producto': prod,
                    'Precio Actual':precioact,
                    'Costo Actual': costoact,
                    'intercepto':safe_round(elasticidad.coeficientes.get('Intercept'), 4),
                    #'Venta Base': safe_round(np.exp(elasticidad.coeficientes.get('Intercept')), 0),
                    'Venta Base': safe_round(
                                np.exp(
                                    (elasticidad.coeficientes.get('Intercept', 0) or 0)
                                    + np.log(float(precioact or 1)) * (elasticidad.coeficientes.get('Precio', 0) or 0)
                                    + (np.log(elasticidad.precio_competencia) * elasticidad.coeficientes.get('PRECIO_COMPETENCIA') if elasticidad.precio_competencia else 0)
                                    + 20 * (elasticidad.coeficientes.get('CLIMA', 0) or 0)
                                ), 
                                0
                            ),

                    #coeficientes
                    'Afectación Precio': safe_round(elasticidad.coeficientes.get('Precio'), 4),
                    'Afectación Clima': safe_round(elasticidad.coeficientes.get('CLIMA'), 4),
                    'Pvalue Intercepto': safe_round(elasticidad.pvalores.get('Intercept'), 4),
                    'Pvalue Precio': safe_round(elasticidad.pvalores.get('Precio'), 4),
                    'Pvalue Clima': safe_round(elasticidad.pvalores.get('CLIMA'), 4),
                    'R cuadrada': safe_round(elasticidad.r2, 3),
                    'Afectación Competencia': safe_round(elasticidad.coeficientes.get('PRECIO_COMPETENCIA'), 4),
                    'Pvalue Competencia': safe_round(elasticidad.pvalores.get('PRECIO_COMPETENCIA'), 4),
                    'Precio Competencia': safe_round(elasticidad.precio_competencia,4),
                    'Nombre Competencia': elasticidad.nombre_competencia
                    #"Insight": insight
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
            af_clima = 0 if pd.isna(af_clima) else af_clima
            r2 = res['R cuadrada']
            precio = res['Precio Actual']
            intercepto = res['intercepto']
            costoact = res['Costo Actual']
            #insight = res['Insight']
            insight = ""
            af_comp = res['Afectación Competencia']
            af_comp = 0 if pd.isna(af_comp) else af_comp
            precio_comp = res['Precio Competencia']
            nombre_comp = res['Nombre Competencia']
 
            
            with st.expander(f" SKU {sku} - {prod} - Canal {res['Canal']}"):

                st.markdown("## Resumen")
                df_sku =df_resultados[['Venta Base','Afectación Precio','Afectación Clima','Pvalue Intercepto','Pvalue Precio','Pvalue Clima','R cuadrada']][df_resultados['SKU']==sku]
                #st.dataframe(df_sku)
                
                st.markdown(f"""
                            📦 **Producto:** {prod}  
                            🆔 **SKU:** {sku}  
                            🏬 **Canal:** {canal}  

                            - 📊 **Ventas base:** {venta_base:,} unidades.  
                            - 💰 **Elasticidad precio:** {af_precio:.2f}.  
                            Esto significa que si el precio aumenta 1%, la venta cambia en aproximadamente **{af_precio:.2f}**%.  
                            
                            """)
                if af_comp != 0:
                    st.markdown(f"""
                    - 💰 **Elasticidad competencia ({nombre_comp}):** {af_comp:.2f}.  
                    Si el precio de la competencia sube 1%, la venta cambia en **{af_comp:.2f}%**.
                    """)

                st.markdown(f"""
                    - 🌦️ **Impacto del clima:** {af_clima:.3f}.  
                    Por cada 1% de incremento en la temperatura el sellout cambia en un **{af_clima:.2%}**.
                    - 📈 **Calidad del modelo (R²):** {r2:.2f}.  
                    El modelo explica un **{r2*100:.2f}**% de la variación de la venta.
                """)
                
                if precio != "":
                    try:
                        precio_actual = float(precio)
                        #intercepto = elasticidad.coeficientes.get('Intercept')
                        #beta_precio = elasticidad.coeficientes.get('Precio')
                        #beta_clima = elasticidad.coeficientes.get('CLIMA')
                        clima_valor = 20  # valor promedio o puedes obtenerlo del layout
                        #st.markdown(intercepto)
                        #st.markdown(af_precio)
                        #st.markdown(af_clima)


                        
                        
                        # Rango de precios (por ejemplo, -20% a +20%)
                        precios = np.arange(precio_actual * 0.9, precio_actual * 1.1 + 0.5, 0.5)

                        # Calcular demanda esperada
                        #demanda = np.exp(intercepto + (np.log(precios) * af_precio) + (np.log(clima_valor) * af_clima))
                        demanda = np.exp(
                                        intercepto
                                        + (np.log(precios) * af_precio)
                                        + (np.log(clima_valor) * af_clima)
                                        + (np.log(precio_comp) * af_comp if precio_comp else 0)
                                    )
                        #+ (np.log(precio_comp) * af_comp if precio_comp else 0)
                        #demanda_df = pd.DataFrame({
                        #    "Precio": precios,
                        #    "Demanda Estimada": demanda,
                        #    "Δ Demanda %": (demanda / demanda[precios == precio_actual][0] - 1) * 100
                        #})

                        #st.markdown("### 📈 Simulación de Demanda vs. Precio")
                        #st.dataframe(demanda_df.style.format({
                        #    "Precio": "{:,.2f}",
                        #    "Demanda Estimada": "{:,.0f}",
                        #    "Δ Demanda %": "{:+.1f}%"
                        #}))
                        idx_precio_actual = (np.abs(precios - precio_actual)).argmin()
                        demanda_df = pd.DataFrame({
                            "Precio": precios,
                            "Demanda Estimada": demanda,
                            "Δ Demanda %": (demanda / demanda[idx_precio_actual] - 1) * 100
                        })

                        # Si se capturó costo, calculamos la utilidad


                        # Verificamos si el costo es un número válido
                        if isinstance(costoact, (int, float)) and not pd.isna(costoact):
                            costo_actual = float(costoact)
                        elif isinstance(costoact, str) and costoact.strip() != "" and costoact.replace(".", "", 1).isdigit():
                            costo_actual = float(costoact)
                        else:
                            costo_actual = None

                        if costo_actual is not None:
                            demanda_df["Utilidad"] = (demanda_df["Demanda Estimada"] * demanda_df["Precio"]) - (
                                demanda_df["Demanda Estimada"] * costo_actual
                            )

                            max_utilidad = demanda_df["Utilidad"].max()

                            def highlight_max(s):
                                is_max = s == s.max()
                                return ["background-color: green" if v else "" for v in is_max]

                            st.markdown("### Simulación de Demanda, Precio y Utilidad")
                            st.dataframe(
                                demanda_df.style
                                .format({
                                    "Precio": "{:,.2f}",
                                    "Demanda Estimada": "{:,.0f}",
                                    "Δ Demanda %": "{:+.1f}%",
                                    "Utilidad": "{:,.2f}",
                                })
                                .apply(highlight_max, subset=["Utilidad"])
                            )
                        else:
                            st.markdown("### 📈 Simulación de Demanda vs. Precio")
                            st.dataframe(
                                demanda_df.style.format({
                                    "Precio": "{:,.2f}",
                                    "Demanda Estimada": "{:,.0f}",
                                    "Δ Demanda %": "{:+.1f}%",
                                })
                            )


                        insight = elasticidad.genera_insight_op(precio=precio,df=demanda_df)
                        col1, col2 = st.columns(2)
                        with col1:
                            # Gráfico interactivo
                            fig_demanda = px.line(
                                demanda_df,
                                x="Precio",
                                y="Demanda Estimada",
                                markers=True,
                                title=f"Curva de Demanda - {prod}",
                            )
                            fig_demanda.add_scatter(
                                x=[precio_actual],
                                y=[demanda[idx_precio_actual]],
                                mode='markers+text',
                                text=["Precio Actual"],
                                textposition="top center",
                                marker=dict(color='red', size=10)
                            )
                            st.plotly_chart(fig_demanda, use_container_width=True)
                            with col2:
                                if sku in graficos_dispersion:
                                    st.plotly_chart(graficos_dispersion[sku], use_container_width=True)
                    except Exception as e:
                        #st.markdown(f"No se pudo generar la simulación de demanda")
                        st.error(f"No se pudo generar la simulación de demanda ({e})")
                else:
                    insight = elasticidad.genera_insight_op()
                    if sku in graficos_dispersion:
                        st.plotly_chart(graficos_dispersion[sku], use_container_width=True)
                    st.info("⚠️ Agrega un precio actual para generar la curva de demanda.")
                
                
                 

                    #st.markdown("")
                #col1, col2 = st.columns([2, 1]) 
               
                if sku in graficos:
                    st.plotly_chart(graficos[sku], use_container_width=True)
               


                
                #st.plotly_chart(fig)
                #st.dataframe(df)
                
                
                st.markdown("## Insight")
                st.markdown(
                            f"""
                            <div style="
                                border: .05px solid gray;   /* Borde gris */
                                padding: 10px;             /* Espacio interno */
                                border-radius: 8px;        /* Esquinas redondeadas */
                                background-color: transparent;  /* Fondo transparente */
                                margin: 10px 0px;          /* Margen arriba y abajo */
                            ">{insight}
                                
                            
                            """,
                            unsafe_allow_html=True
                        )
            
