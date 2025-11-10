import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
from ElasticidadSKU import ElasticidadCB
import numpy as np
import plotly.express as px

# Función para resaltar el precio actual
def highlight_precio_actual(row):
    return ["background-color: red; color: white;" if row["Precio"] == precio_actual else "" for _ in row]

def highlight_max(s):
    is_max = s == s.max()
    return ["background-color: green" if v else "" for v in is_max]





st.set_page_config(page_title="Elasticidades SKU", layout="wide")

st.title("Elasticidades por SKU")
#st.markdown("Sube un layout o captura manualmente los SKUs para calcular elasticidades.")

# Selección de método de carga
#opcion = st.radio("Selecciona cómo quieres cargar los SKUs:", 
#                  ["Subir Layout", "Capturar Manualmente"])

layout = None

#if opcion == "Subir Layout":
#    archivo = st.file_uploader("Sube un archivo CSV o Excel con columnas: SKU, Canal, Clima", 
#                               type=["csv", "xlsx"])
#    if archivo:
#        if archivo.name.endswith(".csv"):
#            layout = pd.read_csv(archivo)
#        else:
#            layout = pd.read_excel(archivo)
#        st.success("Layout cargado correctamente")
#        st.dataframe(layout)

#elif opcion == "Capturar Manualmente":
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
df_productos = pd.read_excel(r'Catálogo Corporativo Final.xlsx',sheet_name='Catálogo')
paisesdisponibles = ['Argentina','México']
df_productos = df_productos[df_productos['Pais'].isin(paisesdisponibles)]
pais = st.selectbox("Pais", sorted(df_productos["Pais"].unique()))
query = f"""  SELECT distinct p.propstcodbarras as SKU,
    p.propstid AS PROPST_ID
    FROM PRD_CNS_MX.DM.FACT_DESPLAZAMIENTOSEMANALCADENASKU AS m
    LEFT JOIN PRD_CNS_MX.DM.VW_DIM_CLIENTE AS c ON m.CteID = c.CteID
    LEFT JOIN PRD_CNS_MX.DM.VW_DIM_PRODUCTO AS p ON m.ProdID = p.ProdID
    LEFT JOIN PRD_CNS_MX.DM.VW_DIM_TIEMPO AS t ON m.TMPID = t.TMPID
    WHERE t.anio >= 2023
        AND c.TIPOESTNOMBRE IN ('Autoservicios','Cadenas de farmacia')
        AND c.TIPOCLIENTE='Monitoreado'"""
query_int =f"""SELECT distinct
            es.PROPSTCODBARRAS as SKU,
            es.propstid AS PROPST_ID
        FROM PRD_CNS_MX.DM.FACT_SO_SEM_CAD_SKU_INT so 
        LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_ESTRUCTURAPRODUCTOSTOTALPAISES es ON es.PROPSTID=so.PROPSTID 
        LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_ESTRUCTURACLIENTESSEGPTVTOTAL cl ON cl.CADID=so.CADID  
        LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_CATSEMANAS s ON s.SEMID=so.SEMID 
        LEFT JOIN PRD_STG.GNM_CT.GNMPAIS p ON p.PAISID=so.PAISID  
        WHERE s.SEMANIO>=2023   
                AND P.PAIS='{pais}'
                AND cl.TIPOESTNOMBRE IN ('Autoservicios','Cadenas de farmacia')
                AND cl.GRPCLASIFICACION='Monitoreado'"""
if pais == 'México':
    df_propstid =  pd.read_sql(query,conn)
    conn.close()
else:
    df_propstid =  pd.read_sql(query_int,conn)
    conn.close()
#df_productos = df_productos[df_productos['IdPais']==1].copy()
df_productos = df_productos[df_productos['Pais']==pais].copy()
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


# Extraemos el SKU limpio y lo dejamos como texto
sku_val_prov = str(sku_row.split(" - ")[0]).strip()   

col1, col2, col3, col4 = st.columns(4)

with col1:
    if pais == 'México':
        canal = st.selectbox("Canal", ["Moderno", "Autoservicios", "Farmacias"])
    if pais == 'Argentina':
        canal = st.selectbox("Canal", ["Autoservicios"])
with col2:
    precio_act = st.text_input("Precio")
with col3:
    # Cargar los costos desde Excel
    if pais == 'México':
        costos = pd.read_excel(r"CostoGestionMensual_2025-10-28-0942 (1).xlsx")
        conn = snowflake.connector.connect(
                user=st.secrets["snowflake"]["user"],
                password=st.secrets["snowflake"]["password"],
                account=st.secrets["snowflake"]["account"],
                database=st.secrets["snowflake"]["database"],
                schema=st.secrets["snowflake"]["schema"]
                )
        query = f"""  SELECT
                    semanio,
                    semmes,
                    fecha,
                    codigobarras,
                    costo_estandar,
                    paisid
                    FROM
                    matriz_precio_costo where paisid=1;"""
        dfcostosmx =  pd.read_sql(query_int,conn)
        conn.close()
        costos = costos.rename(columns={
            'CODIGOBARRAS': 'PROPSTCODBARRAS',
            'COSTO': 'Costo'
        })  
        costos['PROPSTCODBARRAS'] = costos['PROPSTCODBARRAS'].astype(str).str.strip()  
        costos = costos[['PROPSTCODBARRAS', 'Costo']].drop_duplicates()
        # Filtrar por el SKU actual
        costo_filtrado = costos.loc[costos['PROPSTCODBARRAS'] == sku_val_prov, 'Costo']
    else:
        costos = pd.read_excel(r"CostoInternacional_VF.xlsx")

        costos = costos.rename(columns={
            'CODIGOBARRAS': 'PROPSTCODBARRAS',
            'COSTO': 'Costo'
        })
        costos['PROPSTCODBARRAS'] = costos['PROPSTCODBARRAS'].astype(str).str.strip()
        costos['Pais'] = costos['Pais'].str.strip()

        # Filtrar por país y SKU
        costos_filtrados = costos[
            (costos['Pais'] == pais) &
            (costos['PROPSTCODBARRAS'] == sku_val_prov)
        ].copy()

        # Calcular promedio mensual
        costos_mensuales = (
            costos_filtrados
            .groupby(['ANIO', 'MES'], as_index=False)['Costo']
            .mean()
        )

        # Obtener costo más reciente (manteniendo forma de DataFrame)
        costos_mensuales = costos_mensuales.sort_values(['ANIO', 'MES'], ascending=[False, False])

        # Devuelve un DataFrame con una fila (último costo)
        costo_filtrado = costos_mensuales.head(1)


    

    # Si el SKU existe en el archivo, precargar su costo
    if not costo_filtrado.empty:
        costo_default = costo_filtrado.iloc[0]
    else:
        costo_default = ""

    # Mostrar el campo editable con el valor precargado
    costo_act = st.text_input("Costo", value=costo_default)
with col4:  
    #st.markdown("Clima")  
    #clima = st.checkbox("¿Considerar Clima?", value=True)  
    st.markdown("Variables a considerar")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        clima = st.checkbox("Clima", value=False)
    with col2:
        grps = st.checkbox("Grps", value=True)



# AGREGAMOS LA COMPETENCIA SI EXISTE 
col1,col2 = st.columns(2)
with col1:

    # Cargar datos de competencia
    comp = pd.read_excel(r"Competencias_Elasticidades_VF.xlsx")
    comp = comp[comp['Pais']==pais]
    comp.columns = [c.strip() for c in comp.columns]
    comp = comp.rename(columns={
        'SKU': 'PROPSTCODBARRAS',
        'Descripcion Competencia': 'DESC_COMPETENCIA',
        'Precio Competencia': 'PRECIO_COMPETENCIA'
    })
    comp = comp[['PROPSTCODBARRAS','ANIO','DESC_COMPETENCIA','SEMNUMERO','PRECIO_COMPETENCIA']]
    comp['PROPSTCODBARRAS'] = comp['PROPSTCODBARRAS'].astype(str).str.strip()  
    comp = comp[comp['PROPSTCODBARRAS'] == sku_val_prov]  # filtrar por SKU

    # Si hay competencia, permitir seleccionar
    if not comp.empty:
        # Mostrar un selectbox con las descripciones únicas de competidores
        descs_comp = comp['DESC_COMPETENCIA'].unique().tolist()
        #Cambio MLTCOMP
        #seleccion_comp = st.selectbox("Selecciona competencia", options=descs_comp)
        seleccion_comp = st.multiselect("Selecciona una o más competencias", options=descs_comp)

        if seleccion_comp:
            st.write(f"**Competencias seleccionadas:** {', '.join(seleccion_comp)}")
            comp_sel = comp[comp['DESC_COMPETENCIA'].isin(seleccion_comp)]
            # Mostrar precios de las seleccionadas
            ultimos_precios = (
                comp_sel.groupby('DESC_COMPETENCIA')['PRECIO_COMPETENCIA']
                .last()
                .to_dict()
            )

            for nombre, precio in ultimos_precios.items():
                st.write(f"**{nombre}:** {precio:.2f}")
        else:
            st.info("Selecciona al menos una competencia para continuar.")



        # Filtrar la fila seleccionada
        #comp_sel = comp[comp['DESC_COMPETENCIA'] == seleccion_comp]
        #comp_sel = comp[comp['DESC_COMPETENCIA'].isin(seleccion_comp)]


        # Tomar el precio de esa competencia (puedes tomar el último o el promedio)
        #precio_comp = comp_sel['PRECIO_COMPETENCIA'].iloc[-1]  # o .iloc[-1]

        #st.write(f"**Precio competencia seleccionado:** {precio_comp:.2f}")


    else:
        st.info("No hay información de competencia para este SKU.")


# Inicializar lista en session_state
if "manual_layout" not in st.session_state:
    st.session_state.manual_layout = []

# Botón para agregar
if st.button("Agregar SKU a la lista"):
    
    # Verificar si hay variable de competencia seleccionada
    desc_competencia = seleccion_comp if "seleccion_comp" in locals() else ""


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
            "Pais": pais,
            "SKU": sku_val,
            "PropstNombre":prod,
            "Producto base": producto_base,
            "Canal": canal,
            "Clima": clima,
            "Grps": grps,
            "Precio Actual": precio,
            "Costo Actual": costo,
            "DESC_COMPETENCIA": desc_competencia
        })
        st.success(f"SKU {sku_val} agregado a la lista.")
    elif precio_act == "":
        precio =""
        costo =""
        prod = sku_row.split(" - ")[1] 
        sku_val = sku_row.split(" - ")[0]  # extraer el código de barras
        st.session_state.manual_layout.append({
            "Pais": pais,
            "SKU": sku_val,
            "PropstNombre":prod,
            "Producto base": producto_base,
            "Canal": canal,
            "Clima": clima,
            "Grps": grps,
            "Precio Actual": precio,
            "Costo Actual": costo,
            "DESC_COMPETENCIA": desc_competencia
        })
        st.success(f"SKU {sku_val} agregado a la lista.")
    else:
        st.error("⚠️ Solo se permiten números (ej: 123 o 123.45)")
    

# Mostrar tabla acumulada
if st.session_state.manual_layout:
    st.markdown("### SKUs capturados:")
    
    layout = pd.DataFrame(st.session_state.manual_layout)
    layout["DESC_COMPETENCIA"] = layout["DESC_COMPETENCIA"].apply(
        lambda x: x if isinstance(x, list)
        else ([] if pd.isna(x) or x == "" else [x])
    )
    st.dataframe(layout)


# Procesar layout
if layout is not None and st.button("Ejecutar Análisis"):
    resultados = []
    graficos = {}
    graficos_dispersion = {}
    layout = layout.reset_index(drop=True)
    layout["escenario_id"] = layout.index.astype(str)

    with st.spinner("Calculando elasticidades"):
        for _, row in layout.iterrows():
            pais = row['Pais']
            sku = row["SKU"]
            canal = row["Canal"]
            temp = row["Clima"]
            grps = row['Grps']
            prod = row["PropstNombre"]
            precioact = row["Precio Actual"]
            costoact = row['Costo Actual']
            desc_competencia = row['DESC_COMPETENCIA']
            id_escenario = row['escenario_id']
            desc_competencia = row['DESC_COMPETENCIA']
            productobase = row['Producto base']

            
          
            try:
                #elasticidad = ElasticidadCB(codbarras=sku, canal=canal, temp=temp,desc_competencia=desc_competencia)
                elasticidad = ElasticidadCB(
                    codbarras=sku,
                    canal=canal,
                    temp=temp,
                    grps=grps,
                    desc_competencias=desc_competencia,    # ahora es lista, no string
                    pais = pais,
                    productobase = productobase
                )

                elasticidad.consulta_sellout()
                print(elasticidad.status)
                elasticidad.calcula_elasticidad()
                print(elasticidad.status)

                fig = elasticidad.grafica()
                dispersion = elasticidad.grafica_dispersion()
                #graficos[sku] = fig
                #graficos_dispersion[sku] = dispersion
                clave = f"{sku}_{id_escenario}"
                graficos[clave] = fig
                graficos_dispersion[clave] = dispersion
               
                

                
                

                #insight = elasticidad.genera_insight_op()
                def safe_round(value, dec=4):
                    return round(value, dec) if value is not None else None
                # Definir rango de semanas de Julio Regalado
                semanas_JR = list(range(21, 32))  # 21 a 31 inclusive
                semanas_MP = list(range(1, 7))
                
                # Indicador: 1 si está en ese rango, 0 si no
                indicador_JR = 1 if elasticidad.ultima_semana in semanas_JR else 0
                indicador_MP = 1 if elasticidad.ultima_semana in semanas_MP else 0

                competencias_resultados = []
                for col in elasticidad.coeficientes.index:
                    if col.startswith("PRECIO_COMPETENCIA"):
                        competencias_resultados.append({
                            "Nombre Competencia": col.replace("PRECIO_COMPETENCIA_", "").replace("_", " "),
                            "Afectación Competencia": safe_round(elasticidad.coeficientes.get(col), 4),
                            "Pvalue Competencia": safe_round(elasticidad.pvalores.get(col), 4),
                            "Precio Competencia": safe_round(elasticidad.precio_competencia.get(col), 4) if isinstance(elasticidad.precio_competencia, dict) else None
                        })
                #st.markdown("prueba")
                #st.markdown(competencias_resultados)
                #st.markdown(elasticidad.nombre_competencias)
                #st.markdown(elasticidad.status)
                grps = elasticidad.grps
                if grps:
                    grps_actuales = elasticidad.grps_actuales

                else:
                    grps_actuales = 0 
                # Calcular venta base considerando todas las competencias
                comp_effect = 0
                if isinstance(elasticidad.precio_competencia, dict):
                    for col, precio_comp in elasticidad.precio_competencia.items():
                        beta = elasticidad.coeficientes.get(col, 0)
                        if precio_comp and not pd.isna(precio_comp):
                            comp_effect += np.log(precio_comp) * beta
                venta_base = safe_round(
                        np.exp(
                            (elasticidad.coeficientes.get('Intercept', 0) or 0)
                            + np.log(float(precioact or 1)) * (elasticidad.coeficientes.get('Precio', 0) or 0)
                            + comp_effect
                            + 20 * (elasticidad.coeficientes.get('CLIMA', 0) or 0)
                            + grps_actuales *(elasticidad.coeficientes.get('Grps', 0) or 0)
                            + (indicador_JR * elasticidad.coeficientes.get('JULIO_REGALADO', 0) if indicador_JR else 0)
                            + (indicador_MP * elasticidad.coeficientes.get('MEGA_PAUTA', 0) if indicador_MP else 0)
                        ),
                        0
                    )
                
                resultados.append({
                    'Pais':pais,
                    'SKU': sku,
                    'Canal': canal,
                    'Producto': prod,
                    'Temperatura': temp,
                    'Grps':grps,
                    'Precio Actual':precioact,
                    'Costo Actual': costoact,
                    'intercepto':safe_round(elasticidad.coeficientes.get('Intercept'), 4),
                   
                    #'Venta Base': safe_round(np.exp(elasticidad.coeficientes.get('Intercept')), 0),
                    #'Venta Base': safe_round(
                    #            np.exp(
                    #                (elasticidad.coeficientes.get('Intercept', 0) or 0)
                    #                + np.log(float(precioact or 1)) * (elasticidad.coeficientes.get('Precio', 0) or 0)
                    #                + (np.log(elasticidad.precio_competencia) * elasticidad.coeficientes.get('PRECIO_COMPETENCIA') if elasticidad.precio_competencia else 0)
                    #                + 20 * (elasticidad.coeficientes.get('CLIMA', 0) or 0)
                    #                + (indicador_JR*elasticidad.coeficientes.get('JULIO_REGALADO') if indicador_JR else 0)
                    #            ), 
                    #            0
                    #        ),

                    'Venta Base': venta_base,
                    
                    #coeficientes
                    'Afectación Precio': safe_round(elasticidad.coeficientes.get('Precio'), 4),
                    'Afectación Clima': safe_round(elasticidad.coeficientes.get('CLIMA'), 4),
                    'Pvalue Intercepto': safe_round(elasticidad.pvalores.get('Intercept'), 4),
                    'Pvalue Precio': safe_round(elasticidad.pvalores.get('Precio'), 4),
                    'Pvalue Clima': safe_round(elasticidad.pvalores.get('CLIMA'), 4),
                    'R cuadrada': safe_round(elasticidad.r2, 3),
                    #'Afectación Competencia': safe_round(elasticidad.coeficientes.get('PRECIO_COMPETENCIA'), 4),
                    #'Pvalue Competencia': safe_round(elasticidad.pvalores.get('PRECIO_COMPETENCIA'), 4),
                    #'Precio Competencia': safe_round(elasticidad.precio_competencia,4),
                    #'Nombre Competencia': elasticidad.nombre_competencia,

                    'Competencias': competencias_resultados,

                    #Julio Regalado
                    'Pvalue Julio Regalado':safe_round(elasticidad.pvalores.get('JULIO_REGALADO'), 4),
                    'Afectación Julio Regalado':safe_round(elasticidad.coeficientes.get('JULIO_REGALADO'), 4),
                    'Indicador Julio Regalado': indicador_JR,
                    #Mega Pauta
                    'Pvalue Mega Pauta':safe_round(elasticidad.pvalores.get('MEGA_PAUTA'), 4),
                    'Afectación Mega Pauta':safe_round(elasticidad.coeficientes.get('MEGA_PAUTA'), 4),
                    'Indicador Mega Pauta': indicador_MP,

                    #Grps
                    'Pvalue Grps':safe_round(elasticidad.pvalores.get('Grps'), 4),
                    'Afectación Grps':safe_round(elasticidad.coeficientes.get('Grps'), 4),
                    'Grps Actuales': grps_actuales,



                    'Id_unico': id_escenario
                    #"Insight": insight
                })



            except Exception as e:
                st.error(f"Error en SKU {sku}: {e}")

    if resultados:
        df_resultados = pd.DataFrame(resultados)

        

        st.subheader(" Gráficos e Insights por SKU")
        for res in resultados:
            pais = res['Pais']
            escenario_id = res['Id_unico']
            temperatura = res['Temperatura']
            grps = res['Grps']
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
            grps_actuales = res['Grps Actuales']
            af_grps = res['Afectación Grps']
            af_grps = 0 if pd.isna(af_grps) else af_grps
            pv_grps = res['Pvalue Grps']
            #insight = res['Insight']
            insight = ""
            #af_comp = res['Afectación Competencia']
            #af_comp = 0 if pd.isna(af_comp) else af_comp
            #precio_comp = res['Precio Competencia']
            #nombre_comp = res['Nombre Competencia']

            


            #Julio Regalado
            af_JR = res['Afectación Julio Regalado']
            pv_JR = res['Pvalue Julio Regalado']
            af_JR = 0 if pd.isna(af_JR) else af_JR
            indicador_JR = res['Indicador Julio Regalado']


            #Mega Pauta
            af_MP = res['Afectación Mega Pauta']
            pv_MP = res['Pvalue Mega Pauta']
            af_MP = 0 if pd.isna(af_MP) else af_MP
            indicador_MP = res['Indicador Mega Pauta']
        
            
            #st.write(f"Indicador Julio Regalado: {indicador_JR}")
 
            
            with st.expander(f" SKU {sku} - {prod} - Canal {res['Canal']}"):

                st.markdown("## Resumen")
                df_sku =df_resultados[['Venta Base','Afectación Precio','Afectación Clima','Pvalue Intercepto','Pvalue Precio','Pvalue Clima','R cuadrada']][df_resultados['SKU']==sku]
                #st.dataframe(df_sku)
                
                st.markdown(f"""
                            📦 **Producto:** {prod}  
                            🆔 **SKU:** {sku}  
                            🏬 **Canal:** {canal}  

                            - 📊 **Ventas base:** {venta_base:,} unidades.
                            Venta esperada semanal en el canal {canal} dado el precio actual y el promedio del clima y GRPs.
                              
                            """)
                if abs(af_precio) >= 1:
                    st.markdown("**PRODUCTO ELÁSTICO**")
                else:
                    st.markdown("**PRODUCTO INELÁSTICO**")
                
                st.markdown(f"""- 💰 **Elasticidad precio:** {af_precio:.2f}.  
                            Esto significa que si el precio aumenta 1%, la venta cambia en aproximadamente **{af_precio:.2f}**%.""")

                #if af_comp != 0:
                #    st.markdown(f"""
                #    - 💰 **Elasticidad competencia ({nombre_comp}):** {af_comp:.2f}.  
                #    Si el precio de la competencia sube 1%, la venta cambia en **{af_comp:.2f}%**.
                #    """)

               
                if af_JR !=0:
                    if pv_JR <= 0.05:
                        st.markdown(f"""
                        - 📈 **Impacto de promociones Julio Regalado (S21-S31):** {af_JR:.2f}.  
                        Las promociones de Julio Regalado afectan en un **{af_JR:.2f}%** a la venta.
                        """)
                    if pv_JR > 0.05:
                        st.markdown(f"""Promociones Julio Regalado NO tiene importancia estadisticamente (S21-S31)""")


                if af_MP !=0:
                    if pv_MP <= 0.05:
                        st.markdown(f"""
                        - 📈 **Impacto de Mega Pauta (S01-S06):** {af_MP:.2f}.  
                        La Mega Pauta afectan en un **{af_MP:.2f}%** a la venta.
                        """)
                    if pv_MP > 0.05:
                        st.markdown(f"""La Mega Pauta NO tiene importancia estadisticamente (S01-S06)""")
                if temperatura:
                    st.markdown(f"""
                        - 🌦️ **Impacto del clima:** {af_clima:.3f}.  
                        Por cada 1% de incremento en la temperatura el sellout cambia en un **{af_clima:.2f}**%.
                    """)
                if grps:
                    if pv_grps<= 0.05:
                        st.markdown(f"""
                            - 📈 **Grps:** {af_grps:.2f}.  
                            Por cada 1% de incremento en los gprs el sellout cambia en un**{af_grps:.2f}**% .
                            
                        """)
                    else:
                        st.markdown(f"""Los grps NO tienen importancia estadisticamente""")
                st.markdown(f"""
                        - 📈 **Calidad del modelo (R²):** {r2:.2f}.  
                        El modelo explica un **{r2*100:.2f}**% de la variación de la venta.
                    """)
                concatenado_competencia = ""  
                if res.get('Competencias'):
                    st.markdown("💰 **Elasticidades de Competencia**")
                    concatenado_competencia = ""
                    for comp_info in res['Competencias']:
                        nombre_comp = comp_info['Nombre Competencia']
                        concatenado_competencia = concatenado_competencia+nombre_comp
                        af_comp = comp_info['Afectación Competencia']
                        pv_comp = comp_info['Pvalue Competencia']
                        precio_comp = comp_info['Precio Competencia']

                        st.markdown(f"""
                        - **{nombre_comp}**  
                        Si el precio de la competencia sube 1%, la venta cambia en **{af_comp:.2f}%**.
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


                        
                        
                        # Rango de precios 
                        #precios = np.arange(precio_actual * 0.9, precio_actual * 1.1 + 0.5, 0.5)

                        # Generar rango de precios y asegurar que incluya el precio actual exacto
                        precios = np.linspace(precio_actual * 0.85, precio_actual * 1.15, num=31)
                        precios = np.unique(np.append(precios, precio_actual))  # Garantiza que esté el precio exacto
                        precios = np.round(precios, 2)  # Redondea a 2 decimales



                        # Calcular demanda esperada
                        #demanda = np.exp(intercepto + (np.log(precios) * af_precio) + (np.log(clima_valor) * af_clima))
                        #demanda = np.exp(
                        #                intercepto
                        #                + (np.log(precios) * af_precio)
                        #               + (np.log(clima_valor) * af_clima)
                        #                + (np.log(precio_comp) * af_comp if precio_comp else 0)
                        #                + (indicador_JR * af_JR if indicador_JR else 0)
                        #            )


                        # Efecto acumulado de todas las competencias
                        comp_effect = 0
                        if res.get('Competencias'):
                            for comp_info in res['Competencias']:
                                precio_comp = comp_info['Precio Competencia']
                                af_comp = comp_info['Afectación Competencia']
                                if precio_comp and not pd.isna(precio_comp):
                                    comp_effect += np.log(precio_comp) * af_comp
                        if grps:
                            demanda = np.exp(
                                intercepto
                                + (np.log(precios) * af_precio)
                                + (clima_valor * af_clima)
                                + (grps_actuales * af_grps)
                                + comp_effect
                                + (indicador_JR * af_JR if indicador_JR else 0)
                                + (indicador_MP * af_MP if indicador_MP else 0)
                            )
                           
                        else:
                            demanda = np.exp(
                                intercepto
                                + (np.log(precios) * af_precio)
                                + (clima_valor * af_clima)
                                + comp_effect
                                + (indicador_JR * af_JR if indicador_JR else 0)
                                + (indicador_MP * af_MP if indicador_MP else 0)
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
                        demanda_df = demanda_df.drop_duplicates()
                        # Si se capturó costo, calculamos la utilidad


                        # Verificamos si el costo es un número válido
                        if isinstance(costoact, (int, float)) and not pd.isna(costoact):
                            costo_actual = float(costoact)
                        elif isinstance(costoact, str) and costoact.strip() != "" and costoact.replace(".", "", 1).isdigit():
                            costo_actual = float(costoact)
                        else:
                            costo_actual = None

                        if costo_actual is not None:
                            demanda_df["Utilidad"] = (
                                (demanda_df["Demanda Estimada"] * demanda_df["Precio"])
                                - (demanda_df["Demanda Estimada"] * costo_actual)
                            )

                            if demanda_df["Utilidad"].notna().any():
                                max_utilidad = demanda_df["Utilidad"].max()
                            else:
                                st.warning(f"⚠️ No hay utilidades válidas para SKU {sku}. Revisa los datos.")
                                max_utilidad = 0

                            
                            

                            st.markdown("### Simulación de Demanda, Precio y Utilidad")
                            st.dataframe(
                                demanda_df.style
                                .format({
                                    "Precio": "{:,.2f}",
                                    "Demanda Estimada": "{:,.0f}",
                                    "Δ Demanda %": "{:+.1f}%",
                                    "Utilidad": "{:,.2f}",
                                })
                                .apply(highlight_precio_actual, axis=1)
                                .apply(highlight_max, subset=["Utilidad"])
                            )
                        else:
                            st.markdown("### Simulación de Demanda vs. Precio")
                            st.dataframe(
                                demanda_df.style.format({
                                    "Precio": "{:,.2f}",
                                    "Demanda Estimada": "{:,.0f}",
                                    "Δ Demanda %": "{:+.1f}%",
                                })
                                .apply(highlight_precio_actual, axis=1)
                            )


                        insight = elasticidad.genera_insight_op(precio=precio,df=demanda_df)
                        clave = f"{sku}_{escenario_id}"
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
                            st.plotly_chart(fig_demanda, use_container_width=True, key=f"fig_demanda_{sku}_{res['Canal']}_{escenario_id}")
                            with col2:
                                #if sku in graficos_dispersion:
                                #    st.plotly_chart(graficos_dispersion[sku], use_container_width=True, key=f"fig_disp_{sku}_{res['Canal']}_{precio}_{concatenado_competencia}_{escenario_id}")
                                if clave in graficos_dispersion:
                                    st.plotly_chart(
                                        graficos_dispersion[clave],
                                        use_container_width=True,
                                        key=f"fig_disp_{sku}_{res['Canal']}_{escenario_id}"
                                    )

                    except Exception as e:
                        #st.markdown(f"No se pudo generar la simulación de demanda")
                        st.error(f"No se pudo generar la simulación de demanda ({e})")
                else:
                    
                    #if sku in graficos_dispersion:
                    #    st.plotly_chart(graficos_dispersion[sku], use_container_width=True)
                    clave = f"{sku}_{escenario_id}"
                    if clave in graficos_dispersion:
                        st.plotly_chart(
                            graficos_dispersion[clave],
                            use_container_width=True,
                            key=f"fig_disp_{sku}_{res['Canal']}_{escenario_id}"
                        )
                    st.info("⚠️ Agrega un precio actual para generar la curva de demanda.")
                    insight = elasticidad.genera_insight_op(precio=None,df=None)
                
                
                 

                    #st.markdown("")
                #col1, col2 = st.columns([2, 1]) 

                #if sku in graficos:
                #    st.plotly_chart(graficos[sku], use_container_width=True, key=f"fig_base_{sku}_{res['Canal']}_{precio}_{concatenado_competencia}_{escenario_id}")
                
                if clave in graficos:
                    st.plotly_chart(
                        graficos[clave],
                        use_container_width=True,
                        key=f"fig_base_{sku}_{res['Canal']}_{escenario_id}"
                    )
               


                
                #st.plotly_chart(fig)
                #st.dataframe(df)
                
                col1, col2 = st.columns(2)
                with col2:
                    st.image("Logo2.png", width=100) 
                with col1:    
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
            





