import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import snowflake.connector
from ElasticidadSKU import ElasticidadCB
import numpy as np
import plotly.express as px




st.set_page_config(page_title="Elasticidades SKU", layout="wide")

st.title("Elasticidades por SKU")

layout = None

# PRODUCTOS 
conn = snowflake.connector.connect(
    user=st.secrets["snowflake"]["user"],
    password=st.secrets["snowflake"]["password"],
    account=st.secrets["snowflake"]["account"],
    database=st.secrets["snowflake"]["database"],
    schema=st.secrets["snowflake"]["schema"]
    )


df_productos = pd.read_excel(r'Catálogo Corporativo Final.xlsx',sheet_name='Catálogo')
paisesdisponibles = ['Argentina','México','Colombia','Brasil']
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

df_productos = df_productos[df_productos['Pais']==pais].copy()
df_productos = df_propstid.merge(df_productos,left_on='PROPST_ID',right_on='ProPstID',how='left')
df_productos = df_productos.drop_duplicates(subset=['Pais', 'Marca', 'Agrupación Pauta', 'Producto Base', 'SKU'], keep='first')

st.markdown("Agrega un SKU, selecciona canal y clima:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    #marca = st.selectbox("Marca", sorted(df_productos["Marca"].unique()))
    marcas = df_productos["Marca"].dropna().unique()
    marca = st.selectbox("Marca", sorted(marcas))


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


sku_val_prov = str(sku_row.split(" - ")[0]).strip()   

col1, col2, col3, col4= st.columns(4)

with col1:  
    if pais == 'Argentina':
        canal = st.selectbox("Canal", ["Autoservicios"])
    elif pais == 'México':
        canal = st.selectbox("Canal", ["Moderno", "Autoservicios", "Farmacias","Wal-Mart de México"])
    else:
        canal = st.selectbox("Canal", ["Moderno", "Autoservicios", "Farmacias"])
with col2:
    precio_act = st.text_input("Precio")
with col3:
    
    # Cargar los costos desde Excel
    if pais == 'México':
        #query para el futuro
        conn = snowflake.connector.connect(
                user=st.secrets["snowflake"]["user"],
                password=st.secrets["snowflake"]["password"],
                account=st.secrets["snowflake"]["account"],
                database=st.secrets["snowflake"]["database"],
                schema=st.secrets["snowflake"]["schema"]
                )
        query_costos = f"""  SELECT
            semanio AS ANIO,
            semmes AS MES,
            codigobarras,
            costo_estandar AS COSTO_GESTION
            FROM
            PRD_CNS_MX.DM.matriz_precio_costo 
            where paisid=1
            Order By semanio,semmes;"""
        costos =  pd.read_sql(query_costos,conn)
        conn.close()
        #costos = pd.read_excel(r"CostoGestionMensual_2025-10-28-0942 (1).xlsx")

        costos = costos.rename(columns={
            'CODIGOBARRAS': 'PROPSTCODBARRAS',
            'COSTO_GESTION': 'Costo'
        })  
        costos['PROPSTCODBARRAS'] = costos['PROPSTCODBARRAS'].astype(str).str.strip()  
        
        # Filtrar por país y SKU
        costos_filtrados = costos[costos['PROPSTCODBARRAS'] == sku_val_prov].copy()
        

        # Mostrar el campo editable con el valor precargado
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


        if not costo_filtrado.empty:
            # Si es un DataFrame con una columna 'Costo'
            if 'Costo' in costo_filtrado.columns:
                costo_default = costo_filtrado['Costo'].iloc[0]
            else:
                # Si viene de otra forma, tomamos el primer valor
                costo_default = costo_filtrado.iloc[0]
        else:
            costo_default = ""

        costo_act = st.text_input("Costo", value=costo_default)
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
            # Si es un DataFrame con una columna 'Costo'
            if 'Costo' in costo_filtrado.columns:
                costo_default = costo_filtrado['Costo'].iloc[0]
            else:
                # Si viene de otra forma, tomamos el primer valor
                costo_default = costo_filtrado.iloc[0]
        else:
            costo_default = ""

        # Asegurar que el valor sea string (Streamlit lo requiere)
        costo_act = st.text_input("Costo", value=str(costo_default))

with col4:
    variacion_precios = st.text_input("Variacion estimada del precio %: ") 




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

    else:
        st.info("No hay información de competencia para este SKU.")
with col2:  
    st.markdown("Variables a considerar")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        clima = st.checkbox("Clima", value=False)
    with col2:
        grps = st.checkbox("Grps", value=True)

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
        variacion = float(variacion_precios)
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
            "DESC_COMPETENCIA": desc_competencia,
            'Variacion':variacion
        })
        st.success(f"SKU {sku_val} agregado a la lista.")
    elif precio_act == "":
        precio =""
        costo =""
        variacion = ""
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
            "DESC_COMPETENCIA": desc_competencia,
            'Variacion':variacion
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
    st.session_state["layout_df"] = layout  
    st.dataframe(layout)



# Helpers (estilos, highlights)

def highlight_precio_actual(row):
    try:
        if 'Precio' in row.index and 'Precio Actual' in row.index:
            # This function used in styled df in original code; keep simple fallback
            return ['background-color: #e6f7ff' if row['Precio'] == row.get('Precio Actual', None) else '' for _ in row]
        else:
            return ['' for _ in row]
    except Exception:
        return ['' for _ in row]

def highlight_max(s):
    try:
        if s.name == "Utilidad":
            max_idx = s.idxmax()
            return ['font-weight: bold' if i==max_idx else '' for i in range(len(s))]
    except Exception:
        pass
    return ['' for _ in s]


# Inicializar session_state

if "skus_store" not in st.session_state:
    # diccionario clave -> {
    #   "result": res (dict),
    #   "demanda_df": df or None,
    #   "graficos": {"base": fig, "disp": fig, "fe": fig, "demanda": fig},
    #   "elasticidad_obj": elasticidad_instance,
    #   "default_insight": texto_insight
    # }
    st.session_state.skus_store = {}

if "insights_cache" not in st.session_state:
    # opcional: cache de insights generados por clave
    st.session_state.insights_cache = {}


# Layout principal: botón para ejecutar análisis (usa tu layout existente)

st.title("Analítica de Elasticidades")


# Para reproducir la lógica original: botón para lanzar cálculo
if layout is not None and st.button("Ejecutar Análisis"):
    resultados = []
    graficos = {}
    graficos_dispersion = {}
    graficos_FE = {}
    layout = layout.reset_index(drop=True)
    layout["escenario_id"] = layout.index.astype(str)

    def reglas_predeterminadas_pasos_precio(precio_actual, var_precio, metodo="scott_precios"):
        """
        Aplica reglas predeterminadas similares a Sturges/Scott para variación de precios
        
        Métodos disponibles:
        - "sturges_precios": Adaptación de Sturges para pricing
        - "scott_precios": Adaptación de Scott para pricing  
        - "freedman_precios": Adaptación de Freedman-Diaconis para pricing
        - "regla_empirica": Regla empírica de pricing
        """
        
        # Rango total de variación (en unidades monetarias)
        rango_total = precio_actual * (var_precio * 2) / 100
        
        if metodo == "sturges_precios":
            # Adaptación de Sturges: n_bins = 1 + log2(n) → para pricing usamos log del rango
            n_pasos = int(1 + 3.322 * np.log10(rango_total + 1))
            n_pasos = max(5, min(n_pasos, 15))
            
        elif metodo == "scott_precios":
            # Adaptación de Scott: bin_width = 3.5 * σ / n^(1/3)
            # Para pricing: asumimos desviación estándar relativa del 5-15%
            std_relativa = 0.10  # 10% de desviación típica en precios
            std_absoluta = precio_actual * std_relativa
            ancho_paso = 3.5 * std_absoluta / (var_precio ** (1/3))
            n_pasos = int(rango_total / ancho_paso)
            n_pasos = max(6, min(n_pasos, 20))
            
        elif metodo == "freedman_precios":
            # Adaptación de Freedman-Diaconis: bin_width = 2 * IQR / n^(1/3)
            # Para pricing: asumimos IQR relativo del 8-20%
            iqr_relativo = 0.15  # 15% de rango intercuartílico típico
            iqr_absoluto = precio_actual * iqr_relativo
            ancho_paso = 2 * iqr_absoluto / (var_precio ** (1/3))
            n_pasos = int(rango_total / ancho_paso)
            n_pasos = max(7, min(n_pasos, 18))
            
        elif metodo == "regla_empirica":
            # Regla empírica de pricing basada en benchmarks de industria
            if var_precio <= 10:
                n_pasos = 8   # Variación pequeña: alta granularidad
            elif var_precio <= 20:
                n_pasos = 12  # Variación media: granularidad media
            elif var_precio <= 30:
                n_pasos = 15  # Variación grande: granularidad baja
            else:
                n_pasos = 18  # Variación muy grande: mínima granularidad
        
        else:
            # Método por defecto
            n_pasos = 12
        
        return n_pasos

    def metodo_hibrido_inteligente(precio_actual, var_precio, datos_historicos=None):
        """
        Método híbrido que combina reglas predeterminadas con datos históricos
        """
        # Obtener sugerencia de reglas predeterminadas
        n_sturges = reglas_predeterminadas_pasos_precio(precio_actual, var_precio, "sturges_precios")
        n_scott = reglas_predeterminadas_pasos_precio(precio_actual, var_precio, "scott_precios")
        n_empirica = reglas_predeterminadas_pasos_precio(precio_actual, var_precio, "regla_empirica")
        n_freedman = reglas_predeterminadas_pasos_precio(precio_actual, var_precio, "freedman_precios")
        
        # Promediar las sugerencias
        n_sugerido = int(np.mean([n_sturges, n_scott, n_empirica,n_freedman]))
        
        #Ajustar basado en datos históricos si están disponibles
        if datos_historicos is not None and 'Precio' in datos_historicos.columns:
            precios_hist = datos_historicos['Precio'].dropna()
            if len(precios_hist) > 10:
                # Calcular densidad natural de precios
                precios_unicos = len(precios_hist.unique())
                densidad = precios_unicos / len(precios_hist)
                
                # Ajustar según densidad (más densidad → más pasos)
                if densidad > 0.8:  # Alta variabilidad de precios
                    n_sugerido = min(n_sugerido + 3, 20)
                elif densidad < 0.3:  # Baja variabilidad
                    n_sugerido = max(n_sugerido - 2, 8)
        
        #  Asegurar límites razonables
        n_sugerido = max(8, min(n_sugerido, 25))
        
        return n_sugerido

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
            productobase = row['Producto base']
            variacionprecio = row['Variacion']

            try:
                # instancia Elasticidad
                elasticidad = ElasticidadCB(
                    codbarras=sku,
                    canal=canal,
                    temp=temp,
                    grps=grps,
                    desc_competencias=desc_competencia,
                    pais=pais,
                    productobase=productobase
                )

                elasticidad.consulta_sellout()
                elasticidad.calcula_elasticidad()
                fig = elasticidad.grafica()
                dispersion = elasticidad.grafica_dispersion()
                elasticidad.calcula_factor_elastico()
                graf_factor_elastico = elasticidad.grafica_factor_elastico()
               
                #clave = f"{sku}_{id_escenario}"
                def escenario_key(row):
                    parts = [
                        row["SKU"],
                        str(row["Canal"]),
                        f"p{row['Precio Actual']}",
                        f"c{row['Clima']}" if 'Clima' in row else "cNA",
                        f"g{row['Grps']}" if 'Grps' in row else "gNA",
                        f"comp{row['DESC_COMPETENCIA']}",
                        f"pb{row['Producto base']}"
                    ]
                    return "_".join(str(x).replace(" ", "").replace("/", "_") for x in parts)
                clave = escenario_key(row)


                graficos[clave] = fig
                graficos_dispersion[clave] = dispersion
                graficos_FE[clave] = graf_factor_elastico

                def safe_round(value, dec=4):
                    return round(value, dec) if value is not None else None

                semanas_JR = list(range(21, 32))
                indicador_JR = 1 if elasticidad.ultima_semana in semanas_JR else 0

                competencias_resultados = []
                for col in elasticidad.coeficientes.index:
                    if col.startswith("PRECIO_COMPETENCIA"):
                        competencias_resultados.append({
                            "Nombre Competencia": col.replace("PRECIO_COMPETENCIA_", "").replace("_", " "),
                            "Afectación Competencia": safe_round(elasticidad.coeficientes.get(col), 4),
                            "Pvalue Competencia": safe_round(elasticidad.pvalores.get(col), 4),
                            "Precio Competencia": safe_round(elasticidad.precio_competencia.get(col), 4) if isinstance(elasticidad.precio_competencia, dict) else None
                        })

                grps_model = elasticidad.grps
                if grps_model:
                    grps_actuales = elasticidad.grps_actuales
                else:
                    grps_actuales = 0

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
                        + grps_actuales * (elasticidad.coeficientes.get('Grps', 0) or 0)
                        + (indicador_JR * elasticidad.coeficientes.get('JULIO_REGALADO', 0) if indicador_JR else 0)
                    ),
                    0
                )

                res = {
                    'Pais': pais,
                    'SKU': sku,
                    'Canal': canal,
                    'Producto': prod,
                    'Temperatura': temp,
                    'Grps': grps,
                    'Precio Actual': precioact,
                    'Costo Actual': costoact,
                    'Variacion': variacionprecio,
                    'intercepto': safe_round(elasticidad.coeficientes.get('Intercept'), 4),
                    'Venta Base': venta_base,
                    'Afectación Precio': safe_round(elasticidad.coeficientes.get('Precio'), 4),
                    'Afectación Clima': safe_round(elasticidad.coeficientes.get('CLIMA'), 4),
                    'Pvalue Intercepto': safe_round(elasticidad.pvalores.get('Intercept'), 4),
                    'Pvalue Precio': safe_round(elasticidad.pvalores.get('Precio'), 4),
                    'Pvalue Clima': safe_round(elasticidad.pvalores.get('CLIMA'), 4),
                    'R cuadrada': safe_round(elasticidad.r2, 3),
                    'Competencias': competencias_resultados,
                    'Pvalue Julio Regalado': safe_round(elasticidad.pvalores.get('JULIO_REGALADO'), 4),
                    'Afectación Julio Regalado': safe_round(elasticidad.coeficientes.get('JULIO_REGALADO'), 4),
                    'Indicador Julio Regalado': indicador_JR,
                    'Pvalue Grps': safe_round(elasticidad.pvalores.get('Grps'), 4),
                    'Afectación Grps': safe_round(elasticidad.coeficientes.get('Grps'), 4),
                    'Grps Actuales': grps_actuales,
                    'Id_unico': id_escenario,
                    'Variacion':variacionprecio
                }

                # Generar simulación siempre (si hay precio)
                demanda_df = None
                try:
                    precio = precioact
                    af_precio = res['Afectación Precio'] or 0
                    af_clima = res['Afectación Clima'] or 0
                    af_JR = res['Afectación Julio Regalado'] or 0
                    grps_actuales = res['Grps Actuales'] or 0
                    intercepto = res['intercepto'] or 0
                    costoact_local = res['Costo Actual']
                    var_precio = res['Variacion']

                    if precio != "" and precio is not None:
                        precio_actual = float(precio)
                        clima_valor = 20
                        #limit_min = (100-var_precio)/100
                        #limit_max = (100+var_precio)/100
                        #precios = np.linspace(precio_actual * limit_min, precio_actual * limit_max, num=31)
                        #precios = np.unique(np.append(precios, precio_actual))
                        #precios = np.round(precios, 2)

                        def generar_precios_optimizados(precio_actual, var_precio, datos_historicos=None):
                            """
                            Genera array de precios optimizado usando reglas estadísticas
                            """
                            # Determinar número óptimo de pasos
                            n_pasos = metodo_hibrido_inteligente(precio_actual, var_precio, datos_historicos)
                            
                            # Calcular límites
                            limite_inferior = precio_actual * (100 - var_precio) / 100
                            limite_superior = precio_actual * (100 + var_precio) / 100
                            
                            # Generar precios
                            precios = np.linspace(limite_inferior, limite_superior, num=n_pasos)
                            precios = np.round(precios, 2)
                            
                            # Asegurar que el precio actual esté incluido y sea exacto
                            idx_mas_cercano = np.argmin(np.abs(precios - precio_actual))
                            precios[idx_mas_cercano] = precio_actual
                            
                            return precios, n_pasos

                
                        try:
                            # Obtener datos históricos si están disponibles
                            datos_historicos = None
                            if hasattr(elasticidad, 'data_grafico') and elasticidad.data_grafico is not None:
                                datos_historicos = elasticidad.data_grafico
                            
                            # Generar precios optimizados
                            precios, n_pasos = generar_precios_optimizados(
                                precio_actual=float(precio_actual),
                                var_precio=float(var_precio),
                                datos_historicos=datos_historicos
                            )
                            
                            
                            # Mensaje informativo
                            #st.success(f"🎯 **Configuración óptima aplicada**: {n_pasos} puntos de precio calculados usando reglas estadísticas de pricing")
                            
                        except Exception as e:
                            # Fallback al método actual
                            #st.warning(f"⚠️ Usando configuración estándar: {e}")
                            limit_min = (100-var_precio)/100
                            limit_max = (100+var_precio)/100
                            precios = np.linspace(precio_actual * limit_min, precio_actual * limit_max, num=15)
                            precios = np.unique(np.append(precios, precio_actual))
                            precios = np.round(precios, 2)


                        comp_effect_local = 0
                        if res.get('Competencias'):
                            for comp_info in res['Competencias']:
                                precio_comp = comp_info['Precio Competencia']
                                af_comp = comp_info['Afectación Competencia']
                                if precio_comp and not pd.isna(precio_comp):
                                    comp_effect_local += np.log(precio_comp) * af_comp

                        if grps_model:
                            demanda = np.exp(
                                intercepto
                                + (np.log(precios) * af_precio)
                                + (clima_valor * af_clima)
                                + (grps_actuales * (res['Afectación Grps'] or 0))
                                + comp_effect_local
                                + (res['Indicador Julio Regalado'] * af_JR if res['Indicador Julio Regalado'] else 0)
                            )
                        else:
                            demanda = np.exp(
                                intercepto
                                + (np.log(precios) * af_precio)
                                + (clima_valor * af_clima)
                                + comp_effect_local
                                + (res['Indicador Julio Regalado'] * af_JR if res['Indicador Julio Regalado'] else 0)
                            )

                        idx_precio_actual = (np.abs(precios - precio_actual)).argmin()
                        demanda_df = pd.DataFrame({
                            "Precio": precios,
                            "Demanda Estimada": demanda,
                            "Δ Demanda %": (demanda / demanda[idx_precio_actual] - 1) * 100
                        }).drop_duplicates()

                        # calcular utilidad si hay costo válido
                        costo_actual = None
                        if isinstance(costoact_local, (int, float)) and not pd.isna(costoact_local):
                            costo_actual = float(costoact_local)
                        elif isinstance(costoact_local, str) and costoact_local.strip() != "" and costoact_local.replace(".", "", 1).isdigit():
                            costo_actual = float(costoact_local)

                        if costo_actual is not None:
                            demanda_df["Utilidad"] = (
                                (demanda_df["Demanda Estimada"] * demanda_df["Precio"])
                                - (demanda_df["Demanda Estimada"] * costo_actual)
                            )
                except Exception:
                    demanda_df = None

                # Generar insight por defecto (usar método de la instancia si existe)
                #try:
                    #default_insight = elasticidad.genera_insight_op(res, df=demanda_df)
                #except Exception:
                    #default_insight = "No se pudo generar insight por defecto."

                # Guardar todo en session_state
                #clave = f"{sku}_{id_escenario}"
                clave = escenario_key(row)
                st.session_state.skus_store[clave] = {
                    "result": res,
                    "demanda_df": demanda_df,
                    "graficos": {
                        "base": graficos.get(clave),
                        "disp": graficos_dispersion.get(clave),
                        "fe": graficos_FE.get(clave)
                    },
                    "elasticidad_obj": elasticidad,
                    "historico_df":datos_historicos
                    #"default_insight": default_insight
                }

            except Exception as e:
                st.error(f"Error en SKU {sku}: {e}")


# Si no hay datos procesados, avisar

if not st.session_state.skus_store:
    st.info("No hay resultados analíticos disponibles. Pulsa 'Ejecutar Análisis' una vez seleccionados los SKUs.")
    st.stop()

# UI: Selectbox y paneles

#col1_gen, col2_gen = st.columns([5, 3])

# Lista de claves a mostrar (orden natural)
keys_list = list(st.session_state.skus_store.keys())

# Formatea la representación en el selectbox
def format_key(k):
    d = st.session_state.skus_store[k]["result"]
    return f"{d['SKU']} - {d['Producto']} - Canal {d['Canal']} - {d['Id_unico']}"

#with col1_gen:
st.header("Analítica")
seleccion = st.selectbox(
    "Selecciona un SKU / Escenario",
    keys_list,
    format_func=format_key
)

data = st.session_state.skus_store[seleccion]["result"]
demanda_df = st.session_state.skus_store[seleccion].get("demanda_df", None)
grafs = st.session_state.skus_store[seleccion].get("graficos", {})
elasticidad_inst = st.session_state.skus_store[seleccion].get("elasticidad_obj", None)
#default_insight = st.session_state.skus_store[seleccion].get("default_insight", "")

# Mostrar resumen
st.markdown("---")
st.markdown("## Resumen")
venta_base = data.get('Venta Base', 0)
st.markdown(f"""
    📦 **Producto:** {data.get('Producto')}  
    🆔 **SKU:** {data.get('SKU')}  
    🏬 **Canal:** {data.get('Canal')}  

    - 📊 **Ventas base:** {int(venta_base):,} unidades.
    Venta esperada semanal en el canal {data.get('Canal')} dado el precio actual y el promedio del clima y GRPs.
""")

af_precio = data.get('Afectación Precio', 0) or 0
pv_precio = data.get('Pvalue Precio', 1) or 1
if abs(af_precio) >= 1:
    st.info("**PRODUCTO ELÁSTICO**")
else:
    st.info("**PRODUCTO INELÁSTICO**")

st.markdown(f"- 💰 **Elasticidad precio:** {af_precio:.2f}.  Esto significa que si el precio aumenta 1%, la venta cambia en aproximadamente **{af_precio:.2f}**%. Significacia: **{(1-pv_precio)*100:.2f}%**")
#st.markdown(f"")

# Julio Regalado
af_JR = data.get('Afectación Julio Regalado', 0) or 0
pv_JR = data.get('Pvalue Julio Regalado', 1) or 1
if af_JR != 0:
    st.markdown(f"- 📈 **Impacto de promociones Julio Regalado (S21-S31)**. Afectan en **{(np.exp(af_JR)-1)*100:.2f}%** a la venta. Significacia: **{(1-pv_JR)*100:.2f}%**")
    #st.markdown(f"")

# Clima
af_clima = data.get('Afectación Clima', 0) or 0
pv_clima = data.get('Pvalue Clima', 1) or 1
if data.get('Temperatura'):
    st.markdown(f"- 🌦️ **Impacto del clima:** {af_clima*100:.2f}. Por cada 1% de incremento en la temperatura el sellout cambia en un **{af_clima*100:.2f}**%. Significacia: **{(1-pv_clima)*100:.2f}%**")
    #st.markdown(f"")

# Grps
if data.get('Grps'):
    af_grps = data.get('Afectación Grps', 0) or 0
    pv_grps = data.get('Pvalue Grps', 1) or 1
    st.markdown(f"- 📈 **Grps:** {af_grps*100:.2f}. Por cada 1% de incremento en los GRPs el sellout cambia en un **{af_grps*100:.2f}**%. Significacia: **{(1-pv_grps)*100:.2f}%**")
    #st.markdown(f"")

# Calidad del modelo
r2 = data.get('R cuadrada', 0) or 0
st.markdown(f"- 📈 **Calidad del modelo (R²):** {r2:.2f}. El modelo explica un **{r2*100:.2f}%** de la variación de la venta.")
if r2 < 0.25:
    st.info("**AJUSTE BAJO**")
elif r2 >= 0.65:
    st.info("**AJUSTE ALTO**")
else:
    st.info("**AJUSTE MODERADO**")

# Elasticidades de competencia
if data.get('Competencias'):
    st.markdown("💰 **Elasticidades de Competencia**")
    for comp_info in data['Competencias']:
        nombre_comp = comp_info['Nombre Competencia']
        af_comp = comp_info['Afectación Competencia'] or 0
        pv_comp = comp_info['Pvalue Competencia'] or 1
        st.markdown(f"- **{nombre_comp}**: si el precio sube 1%, la venta cambia en **{af_comp:.2f}%**. Significacia: **{(1-pv_comp)*100:.2f}%**")

# Mostrar gráfica de demanda si existe
if demanda_df is not None and not demanda_df.empty:
    def highlight_precio_actual(row):
        return [
            "background-color: red; color: white;"
            if row["Precio"] == precio_actual else ""
            for _ in row
        ]

    def highlight_max(s):
        is_max = s == s.max()
        return ["background-color: green; color: white;" if v else "" for v in is_max]
    st.markdown("---")
    
    # Mostrar tabla con estilos (streamlit acepta .style but will render as static table)
    try:
        precio_actual = float(data.get('Precio Actual'))

        # Aplicar estilos
        styled_df = demanda_df.style.format({
            "Precio": "{:,.2f}",
            "Demanda Estimada": "{:,.0f}",
            "Δ Demanda %": "{:+.1f}%",
            "Utilidad": "{:,.2f}"
        }).apply(highlight_precio_actual, axis=1) \
        .apply(highlight_max, subset=["Utilidad"])
        colaux1,colaux2,colaux3 = st.columns([1,4,1])
        with colaux2:
            st.markdown("### Simulación de Demanda, Precio y Utilidad")
            st.dataframe(styled_df, use_container_width=True)
    except Exception:
        st.dataframe(demanda_df)

    # Gráfico de curva de demanda
    try:
        precio_actual_val = float(data.get('Precio Actual'))
        idx = (np.abs(demanda_df["Precio"] - precio_actual_val)).argmin()

        fig_demanda = px.line(
            demanda_df,
            x="Precio",
            y="Demanda Estimada",
            markers=True,
            title=f"Curva de Demanda - {data.get('Producto')}",
        )

        # punto rojo
        fig_demanda.add_scatter(
            x=[precio_actual_val],
            y=[demanda_df.loc[idx, "Demanda Estimada"]],
            mode="markers+text",
            marker=dict(color="red", size=12),
            text=["Precio Actual"],
            textposition="top center"
        )
        colaux1,colaux2,colaux3 = st.columns([1,4,1])
        with colaux2:
            st.plotly_chart(fig_demanda, use_container_width=True, key=f"fig_demanda_{seleccion}")
    except Exception as e:
        st.warning(f"No se pudo generar gráfico de demanda: {e}")

else:
    st.info("⚠️ No hay precio válido para generar la simulación de demanda.")

# Mostrar gráficos que generaste originalmente (base, dispersion, factor elastico)
if grafs.get("disp") is not None:
    try:
        colaux1,colaux2,colaux3 = st.columns([1,4,1])
        with colaux2:
            st.plotly_chart(grafs["disp"], use_container_width=True, key=f"fig_disp_{seleccion}")
    except Exception:
        pass
#if grafs.get("fe") is not None:
#    try:
#        st.plotly_chart(grafs["fe"], use_container_width=True, key=f"fig_fe_{seleccion}")
#    except Exception:
#        pass


# Agregar esta sección después de mostrar el gráfico de factor elástico
if grafs.get("fe") is not None:
    try:
        colaux1,colaux2,colaux3 = st.columns([1,4,1])
        with colaux2:
            st.plotly_chart(grafs["fe"], use_container_width=True, key=f"fig_fe_{seleccion}")
        
        
    except Exception as e:
        st.error(f"Error al mostrar la calculadora de precios: {e}")


    if grafs.get("base") is not None:
        try:
            st.markdown("---")
            st.plotly_chart(grafs["base"], use_container_width=True, key=f"fig_base_{seleccion}")
        except Exception:
            pass

#with col2_gen:
st.markdown("---")
col1, col2,col3 = st.columns([1, 1,20])

with col1:
    st.image("LogoGemini.png", width=50) 
    
    
with col2:
    st.header("IA")

st.subheader("Insight por SKU")

#RECOMENDACION DE PRECIO

if "reco_cache" not in st.session_state:
    st.session_state.reco_cache = {}

reco_key = f"gen_reco_{seleccion}"

if st.button("Recomendacion de precio", key=reco_key):
    if seleccion in st.session_state.reco_cache:
        st.info("La recomendación ya fue generada")
    else:
        try:
            df_hist = st.session_state.skus_store[seleccion]["historico_df"]
            elasticidad_inst = st.session_state.skus_store[seleccion]["elasticidad_obj"]
            demanda_local = st.session_state.skus_store[seleccion].get("demanda_df")
            res_local = st.session_state.skus_store[seleccion].get("result")

            reco = elasticidad_inst.generar_recomendacion(
                df_hist,
                res_local,
                df=demanda_local
            )

            st.session_state.reco_cache[seleccion] = reco
            st.success("Recomendación generada y guardada")

        except Exception as e:
            st.error(f"Error al generar recomendación: {e}")
if seleccion in st.session_state.reco_cache:
    st.markdown("**PRECIO RECOMENDADO**")
    st.markdown(
        f"<div style='border: .05px solid gray; padding: 8px; border-radius:6px'>"
        f"{st.session_state.reco_cache[seleccion]}</div>",
        unsafe_allow_html=True
    )




# Recuperar insight ya generado (si existe)
if st.session_state.insights_cache.get(seleccion):
    st.markdown("**Insight previamente generado**")
    st.markdown(
        f"<div style='border: .05px solid gray; padding: 8px; border-radius:6px, background-color: #f2f2f2; color: #6e6e6e;',>{st.session_state.insights_cache[seleccion]}</div>",
        unsafe_allow_html=True
    )


complemento = st.text_area("Complemento (opcional)", placeholder="Agregar una pregunta adicional...")

gen_key = f"gen_insight_{seleccion}"
if st.button("Generar Insight", key=gen_key):
    try:


        
        #df_hist = st.session_state.skus_store[seleccion]["historico_df"]

       

        elasticidad_inst = st.session_state.skus_store[seleccion].get("elasticidad_obj", None)
        demanda_local = st.session_state.skus_store[seleccion].get("demanda_df", None)
        res_local = st.session_state.skus_store[seleccion].get("result")

        generated = elasticidad_inst.genera_insight_op(res_local, df=demanda_local, complemento=complemento)
        generated = str(generated)

        #reco = elasticidad_inst.generar_recomendacion(df_hist,res_local, df=demanda_local, complemento=complemento)
        #st.markdown("**PRECIO RECOMENDADO**")
        #st.markdown(
        #    f"<div style='border: .05px solid gray; padding: 8px; border-radius:6px'>{reco}</div>",
        #    unsafe_allow_html=True
        #)
        st.session_state.insights_cache[seleccion] = generated
        
        

        st.success("Insight generado y guardado")
        # Si ya existe insight generado, lo mostramos
        if st.session_state.insights_cache.get(seleccion):
            st.markdown("**Insight generado**")
            st.markdown(
                f"<div style='border: .05px solid gray; padding: 8px; border-radius:6px'>{st.session_state.insights_cache[seleccion]}</div>",
                unsafe_allow_html=True
            )


    except Exception as e:
        st.error(f"Error al generar insight: {e}")

