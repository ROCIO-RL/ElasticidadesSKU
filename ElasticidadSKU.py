# LIBRERIAS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import statsmodels.formula.api as smf
from snowflake.connector.pandas_tools import write_pandas
import snowflake.connector
from langchain.prompts import PromptTemplate
from sklearn.linear_model import LinearRegression
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
pd.options.display.float_format = '{:,.2f}'.format
import os
import plotly.graph_objects as go
from openai import OpenAI
import streamlit as st



# CLIMA
conn = snowflake.connector.connect(
    user=st.secrets["snowflake"]["user"],
    password=st.secrets["snowflake"]["password"],
    account=st.secrets["snowflake"]["account"],
    database=st.secrets["snowflake"]["database"],
    schema=st.secrets["snowflake"]["schema"]
    )

query = f"""SELECT 
            TMPANIOSEMANAGENOMMA, 
            TMPSEMANAANIOGENOMMA, 
            TEMPMAX 
        FROM PRD_CNS_MX.SO_HECHOS.VW_DATOS_CLIMA_MX 
        WHERE TMPANIOSEMANAGENOMMA>=2023"""
query2="""SELECT 
            PROPSTCODBARRAS,
            MIN(PROPSTID) AS PROPSTID 
        FROM PRD_CNS_MX.DM.VW_DIM_PRODUCTO 
        GROUP BY PROPSTCODBARRAS"""
clima_bd = pd.read_sql(query,conn)
clima_bd.columns=['Año','Sem','Temperatura']
codbarras=pd.read_sql(query2,conn)
conn.close()


# CLASE ELASTICIDAD
class ElasticidadCB:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np 
    import statsmodels.formula.api as smf
    from snowflake.connector.pandas_tools import write_pandas
    import snowflake.connector
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama3")
    pd.options.display.float_format = '{:,.2f}'.format
    def __init__(self, codbarras, canal, temp,desc_competencia,ruta_competencia="Competencia_Elasticidades.xlsx"):
        """
        codbarras: Código de barras del producto
        canal: 'Autoservicios', 'Farmacias' o 'Moderno'
        temp: True/False si se desea incluir clima
        """
        self.codbarras = codbarras
        self.canal = canal
        self.temp = temp
        self.ruta_competencia = ruta_competencia
        self.precio_competencia = None 
        self.nombre_competencia = desc_competencia

    def calcula_precio(self, venta):
        # Filtrado según canal
        if self.canal == 'Autoservicios':
            venta = venta[(venta['CADID'].isin([2,1,15,18,3,593])) &
                          (venta['MONTORETAIL'] > 0) &
                          (venta['UNIDADESDESP'] > 0)].copy()
        elif self.canal == 'Farmacias':
            venta = venta[(venta['CADID'].isin([27,29])) &
                          (venta['MONTORETAIL'] > 0) &
                          (venta['UNIDADESDESP'] > 0)].copy()
        elif self.canal == 'Moderno':
            venta = venta[(venta['CADID'].isin([1,27,18,15,2,16,3,593])) &
                          (venta['MONTORETAIL'] > 0) &
                          (venta['UNIDADESDESP'] > 0)].copy()

        # Precio unitario
        venta['Precio'] = venta['MONTORETAIL'] / venta['UNIDADESDESP']
        # Aplicar IVA
        tasa_iva = 0.16
        venta['Precio'] = venta['Precio'] * (1 + tasa_iva)

        # Promedio semanal
        if self.canal in ['Autoservicios','Moderno']:
            clientes_por_semana = venta.groupby(['ANIO','SEMNUMERO'])['CADID'].nunique().reset_index()
            semanas_validas = clientes_por_semana[clientes_por_semana['CADID'] >= 3][['ANIO','SEMNUMERO']]
            venta = venta.merge(semanas_validas, on=['ANIO','SEMNUMERO'])
        
        precio = venta.groupby(['ANIO','SEMNUMERO'])['Precio'].mean().reset_index()
        return precio

    def consulta_sellout(self):
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"]
        )

        query = f"""
        SELECT t.ANIO, t.SEMNUMERO, p.propstcodbarras, c.cadid,
               m.UnidadesDesp, NVL(m.MontoDesp,0) AS MontoDespNeto,
               NVL(m.montodespcte,0) AS MontoRetail
        FROM PRD_CNS_MX.DM.FACT_DESPLAZAMIENTOSEMANALCADENASKU AS m
        LEFT JOIN PRD_CNS_MX.DM.VW_DIM_CLIENTE AS c ON m.CteID = c.CteID
        LEFT JOIN PRD_CNS_MX.DM.VW_DIM_PRODUCTO AS p ON m.ProdID = p.ProdID
        LEFT JOIN PRD_CNS_MX.DM.VW_DIM_TIEMPO AS t ON m.TMPID = t.TMPID
        WHERE t.anio >= 2023
          AND p.propstcodbarras = '{self.codbarras}'
          AND c.TIPOESTNOMBRE IN ('Autoservicios','Cadenas de farmacia')
          AND c.TIPOCLIENTE='Monitoreado'
        """
        self.sellout = pd.read_sql(query, conn)
        conn.close()

        # Filtro de cadenas según canal
        if self.canal == 'Autoservicios':
            self.sellout = self.sellout[self.sellout['CADID'].isin([1,10,100,102,15,16,18,19,2,20,21,25,3,342,380,4,5,593,652,11,12,13,381,493,6,9])]
        elif self.canal == 'Farmacias':
            self.sellout = self.sellout[~self.sellout['CADID'].isin([1,10,100,102,15,16,18,19,2,20,21,25,3,342,380,4,5,593,652,11,12,13,381,493,6,9])]

        # Precios semanales
        self.precio_gli = self.calcula_precio(self.sellout)

        # Ventas semanales + precio promedio
        self.sellout = self.sellout.groupby(['ANIO','SEMNUMERO']).agg({'UNIDADESDESP':'sum'}).reset_index()
        self.sellout = self.sellout.merge(self.precio_gli, on=['ANIO','SEMNUMERO'], how='left')

        return self.sellout
    
    def carga_competencia(self):
        try:
            comp = pd.read_excel(r"Competencias_Elasticidades.xlsx")
            comp.columns = [c.strip() for c in comp.columns]
            comp = comp.rename(columns={
                'SKU': 'PROPSTCODBARRAS',
                'Descripcion Competencia': 'DESC_COMPETENCIA',
                'Precio Competencia': 'PRECIO_COMPETENCIA'
            })
            comp = comp[['PROPSTCODBARRAS','ANIO','DESC_COMPETENCIA','SEMNUMERO','PRECIO_COMPETENCIA']]
            comp = comp[comp['PROPSTCODBARRAS'] == self.codbarras]  # filtrar por el SKU actual
            nombre_comp = self.nombre_competencia
            #primera_desc = comp['DESC_COMPETENCIA'].iloc[1]
            # Filtrar todas las filas que tengan esa misma descripción
            #comp = comp[comp['DESC_COMPETENCIA'] == primera_desc]
            if not comp.empty:
                # Tomar la primera descripción disponible
                #primera_desc = comp['DESC_COMPETENCIA'].iloc[0]
                #primera_desc = self.nombre_competencia 
                #comp['DESC_COMPETENCIA']=comp['DESC_COMPETENCIA'].astype(str).str.strip() 

                # Convertir a string y limpiar espacios
                #comp['DESC_COMPETENCIA'] = comp['DESC_COMPETENCIA'].astype(str).strip()
                  # limpia el valor de la clase

                # Filtrar
                comp_filtrado = comp[comp['DESC_COMPETENCIA'] == nombre_comp]

                # Filtrar todas las filas que tengan esa descripción
                #comp = comp[comp['DESC_COMPETENCIA'] == primera_desc]
                comp_filtrado = comp_filtrado[['PROPSTCODBARRAS','ANIO','SEMNUMERO','PRECIO_COMPETENCIA']]
            print(comp_filtrado)
            return comp_filtrado
        except Exception as e:
            print(f"No se pudo cargar competencia: {e}")
            return pd.DataFrame()


    def prepara_datos(self):
        layout = self.sellout[(self.sellout['UNIDADESDESP'] > 0) & (self.sellout['Precio'].notna())].copy()
        layout = layout[layout['ANIO'] >= 2023]

        print(f"Valores nulos:\n{layout.isna().sum()}\nBorrando nulos...")

        if layout['Precio'].isna().sum() > 30:
            raise ValueError("El dataframe contiene demasiados nulos")

        layout.dropna(inplace=True)

        if self.temp:
            temperatura = clima_bd.copy()
            temperatura.columns = ['ANIO','SEMNUMERO','CLIMA']
            layout = layout.merge(temperatura, on=['ANIO','SEMNUMERO'], how='left')

        # Competencia
        competencia = self.carga_competencia()
        if not competencia.empty:
            competencia = competencia.sort_values(['ANIO','SEMNUMERO'], ascending=[True, True])
            self.precio_competencia = float(competencia['PRECIO_COMPETENCIA'].iloc[-1])
            layout = layout.merge(competencia, 
                                left_on=['ANIO','SEMNUMERO'], 
                                right_on=['ANIO','SEMNUMERO'], 
                                how='left')

            if 'PRECIO_COMPETENCIA' in layout.columns and layout['PRECIO_COMPETENCIA'].notna().sum() > 0:
                layout['PRECIO_COMPETENCIA'] = layout['PRECIO_COMPETENCIA'].astype(float)
                #self.precio_competencia = float(layout['PRECIO_COMPETENCIA'].iloc[-1])
                print("Información de competencia agregada correctamente.")
            else:
                self.precio_competencia = None
                print("No hay precios de competencia válidos.")
        else:
            print("No se encontró información de competencia para este SKU.")

        layout_log = layout.copy()
        self.data_grafico = layout.copy()

        # log-log
        layout_log[['UNIDADESDESP','Precio']] = layout_log[['UNIDADESDESP','Precio']].apply(np.log)
        # competencia
        if 'PRECIO_COMPETENCIA' in layout_log.columns and layout_log['PRECIO_COMPETENCIA'].notna().sum() > 0:
            layout_log['PRECIO_COMPETENCIA'] = np.log(layout_log['PRECIO_COMPETENCIA'])

        return layout_log

    def calcula_elasticidad(self):
        data = self.prepara_datos()

        if data.shape[0] < 30:
            raise ValueError("No hay datos suficientes para el modelo")

        """if self.temp:
            modelo = smf.ols('UNIDADESDESP ~ Precio + CLIMA', data=data).fit()
        else:
            modelo = smf.ols('UNIDADESDESP ~ Precio', data=data).fit()"""
        
        formula = 'UNIDADESDESP ~ Precio'

        if self.temp:
            formula += ' + CLIMA'

        if 'PRECIO_COMPETENCIA' in data.columns and data['PRECIO_COMPETENCIA'].notna().sum() > 0:
            formula += ' + PRECIO_COMPETENCIA'

        print(f"Fórmula del modelo: {formula}")
        modelo = smf.ols(formula, data=data).fit()

        print(modelo.summary())
        self.r2 = modelo.rsquared
        self.coeficientes = modelo.params
        self.pvalores = modelo.pvalues
    


    def grafica(self):
        df = self.data_grafico.copy()

        # Crear columna de fecha
        df['Fecha'] = pd.to_datetime(
            df['ANIO'].astype(str) +
            df['SEMNUMERO'].astype(str).str.zfill(2) + '1', 
            format='%G%V%u'
        )

        # Eje de barras (unidades)
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df['Fecha'],
            y=df['UNIDADESDESP'],
            name='Unidades',
            marker_color='lightgray',
            yaxis='y1'
        ))

        # Eje de línea (precio)
        """fig.add_trace(go.Scatter(
            x=df['Fecha'],
            y=df['Precio'],
            name='Precio GLI',
            #mode='lines+markers',
            mode='lines',
            line=dict(color='red',width=2),
            yaxis='y2'
        ))"""
        # Eje de línea (precio propio)
        fig.add_trace(go.Scatter(
            x=df['Fecha'],
            y=df['Precio'],
            name='Precio GLI',
            mode='lines',
            line=dict(color='red',width=2),
            yaxis='y2'
        ))

        # Eje de línea (precio competencia)
        if 'PRECIO_COMPETENCIA' in df.columns and df['PRECIO_COMPETENCIA'].notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=df['Fecha'],
                y=df['PRECIO_COMPETENCIA'],
                name='Precio Competencia',
                mode='lines',
                line=dict(color='blue', width=2, dash='dot'),
                yaxis='y2'
            ))
                

        # Eje clima (si existe)
        if 'CLIMA' in df.columns:
            temp_norm = (df['CLIMA'] - df['CLIMA'].min()) / \
                        (df['CLIMA'].max() - df['CLIMA'].min())
            fig.add_trace(go.Scatter(
                x=df['Fecha'],
                y=temp_norm,
                fill='tozeroy',
                mode='none',
                name='Clima (normalizada)',
                fillcolor='rgba(0, 160, 220, 0.2)',
                yaxis='y3'
            ))

        # Layout con múltiples ejes
        fig.update_layout(
            title="Unidades vendidas vs Precio propio y Clima",
            xaxis=dict(title="Semana"),
            yaxis=dict(title="Unidades", side='left', showgrid=False),
            yaxis2=dict(title="Precio", overlaying='y', side='right'),
            yaxis3=dict(title="Clima (escala visual)", 
                        overlaying='y', side='right', position=0.95, showgrid=False),
            bargap=0.2,
            legend=dict(orientation='h', yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            hovermode="x unified",
            template="plotly_white",
            height=600
        )

        return fig

    def grafica_dispersion(self):
        df = self.data_grafico.copy()

        if 'UNIDADESDESP' not in df.columns or 'Precio' not in df.columns:
            raise ValueError("El DataFrame debe contener las columnas 'UNIDADESDESP' y 'Precio'.")

        X = df['Precio'].values.reshape(-1, 1)
        y = df['UNIDADESDESP'].values

        # Modelo de tendencia
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        fig = go.Figure()

        # Scatter
        fig.add_trace(go.Scatter(
            x=df['Precio'],
            y=df['UNIDADESDESP'],
            mode='markers',
            name='Datos',
            marker=dict(color='royalblue', size=8, line=dict(width=0.5, color='gray')),
            hovertemplate="Precio: %{x}<br>Unidades: %{y}<extra></extra>"
        ))

        # Línea de tendencia
        fig.add_trace(go.Scatter(
            x=df['Precio'],
            y=y_pred,
            mode='lines',
            name='Tendencia',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title="Dispersión: Precio vs Ventas",
            xaxis_title="Precio",
            yaxis_title="Unidades vendidas",
            template="plotly_white",
            height=500
        )

        return fig


    

    def grafico_demanda(self, precio_actual, variable_precio, pasos_atras=5, pasos_adelante=5, incremento=5, otras_vars=None):
        """
        Genera un gráfico interactivo y una tabla de demanda estimada alrededor de un precio actual.

        Parámetros:
        ----------
        - precio_actual: float
            Precio base desde donde partirán las simulaciones.
        - variable_precio: str
            Nombre de la variable de precio principal que se va a variar (por ej. 'XL-3 XTRA 12').
        - pasos_atras: int
            Número de pasos hacia precios menores.
        - pasos_adelante: int
            Número de pasos hacia precios mayores.
        - incremento: float
            Valor de cada salto en el precio.
        - otras_vars: dict
            Diccionario con valores fijos para las otras variables (por ejemplo: {'Clima': 20, 'Tabcin Active 12': 75})

        Retorna:
        -------
        - fig: objeto Plotly Figure
        - df_pred: DataFrame con precios y demanda estimada
        """

        if otras_vars is None:
            otras_vars = {}

        # Lista de precios a simular
        precios = [precio_actual + i*incremento for i in range(-pasos_atras, pasos_adelante+1)]

        # Inicializamos la tabla
        demanda_estim = []

        for precio in precios:
            # Comenzamos con el intercepto
            valor_lineal = self.coeficientes.get('Intercept', 0)

            for var, coef in self.coeficientes.items():
                if var == 'Intercept':
                    continue

                # Usar el precio actual si es la variable que estamos variando
                if var == variable_precio:
                    valor_variable = precio
                    valor_lineal += np.log(valor_variable) * coef
                else:
                    # Tomar de otras_vars o por defecto 1
                    valor_variable = otras_vars.get(var, 1)
                    # Si el nombre sugiere que es un precio, usar log
                    if isinstance(valor_variable, (int, float)) and valor_variable > 0:
                        valor_lineal += np.log(valor_variable) * coef
                    else:
                        valor_lineal += valor_variable * coef

            # Convertimos la suma lineal en demanda
            demanda = np.exp(valor_lineal)
            demanda_estim.append(demanda)

        # Crear DataFrame de resultados
        df_pred = pd.DataFrame({
            'Precio': precios,
            'Demanda Estimada': demanda_estim
        })

        # Crear gráfico interactivo
        fig = go.Figure()

        # Curva de demanda
        fig.add_trace(go.Scatter(
            x=df_pred['Precio'],
            y=df_pred['Demanda Estimada'],
            mode='lines+markers',
            name='Demanda estimada',
            marker=dict(color='royalblue', size=8)
        ))

        # Punto del precio actual
        fig.add_trace(go.Scatter(
            x=[precio_actual],
            y=[df_pred.loc[df_pred['Precio']==precio_actual, 'Demanda Estimada'].values[0]],
            mode='markers',
            name='Precio Actual',
            marker=dict(color='red', size=10, symbol='diamond')
        ))

        fig.update_layout(
            title=f"Estimación de Demanda vs Precio ({variable_precio})",
            xaxis_title="Precio",
            yaxis_title="Demanda estimada",
            template="plotly_white",
            height=600
        )

        return fig, df_pred
        
    def genera_insight(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
            if not hasattr(self, 'r2') or not hasattr(self, 'coeficientes') or not hasattr(self, 'pvalores'):
                raise ValueError("Ejecuta .calcula_elasticidad() antes de generar el insight.")

            
            coef_pval = "\n".join(
                f"- {var}: coef = {self.coeficientes[var]:.4f}, p = {self.pvalores[var]:.4g}"
                for var in self.coeficientes.index
            )

            template = f"""Eres un analista mexicano experto en econometría. 
                            Has corrido un modelo log-log de elasticidad de precios para un SKU.

                            Resultados del modelo:
                            - R²: {self.r2:.4f}
                            - Coeficientes y p-values: {coef_pval}

                            Tu tarea:
                            - Responde en español.
                            - Sé ejecutivo, breve y claro; tu audiencia puede no tener conocimiento técnico de regresiones.
                            - Usa viñetas claras y lenguaje comprensible.
                            - Enfócate en conclusiones de elasticidad y cómo afectan el negocio.
                            - Explica cómo un incremento en el precio o cambios en el clima impactan las unidades vendidas.
                            - Recuerda que los resultados están en **escala logarítmica**; para interpretar en unidades, transforma con e^(coeficiente) para el intercepto.

                            Incluye en tu análisis:
                            1. Variables **significativas** (p-value < 0.05).
                            2. Variable con **mayor impacto** sobre las ventas.
                            3. **Calidad del ajuste** (R²) de manera comprensible.
                            4. **Implicaciones estratégicas** para precios y clima.

                            Prioriza conclusiones accionables para la toma de decisiones, no detalles técnicos innecesarios.

                          """


            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=st.secrets["HUGGINGFACE"]["HF_TOKEN"], 
            )

            # Llamada al modelo
            completion = client.chat.completions.create(
                model=model_name,  
                messages=[{"role": "user", "content": template}],
                temperature=0.3 
            )

            resultado = completion.choices[0].message.content.strip()

            print(resultado)
            return resultado
    

    def genera_insight_op(self, model_name="deepseek-ai/DeepSeek-V3.1-Terminus",precio=None,df=None):
        #DeepSeek-V3.1-Terminus
        #DeepSeek-V3.1-Base
        if not hasattr(self, 'r2') or not hasattr(self, 'coeficientes') or not hasattr(self, 'pvalores'):
            raise ValueError("Ejecuta .calcula_elasticidad() antes de generar el insight.")

        coef_pval = "\n".join(
            f"- {var}: coef = {self.coeficientes[var]:.4f}, p = {self.pvalores[var]:.4g}"
            for var in self.coeficientes.index
        )

        template = f"""Contexto: Eres un Econometrista Senior que se especializa en transformar resultados estadísticos complejos en recomendaciones estratégicas claras y accionables para la alta dirección. Tu estilo es directo, ejecutivo y basado en datos.

        Instrucciones para tu Respuesta:

        Formato: Usa solo viñetas (-). No incluyas introducciones, conclusiones o texto de relleno.

        Lenguaje: Exclusivamente en español. Evita toda jerga técnica innecesaria. Si usas un término técnico (como "p-value"), explícalo brevemente entre paréntesis de inmediato.

        Extensión: Máximo 6 viñetas en total. Sé brutalmente conciso.

        Tarea 1: Recomendación de Precio (Si los datos están disponibles)

        Si se proporciona la tabla de demanda{df} y el precio actual{precio}, procede:

        - Precio Ideal Propuesto: [Precio específico].

        - Rango Alternativo: [Rango de precios, ej. $X - $Y]. Ventaja: [Beneficio clave, ej. 'mejora el margen sin perder volumen significativo'].

        Tarea 2: Análisis del Modelo Log-Log (Siempre requerido)
        Analiza los resultados proporcionados 
        - R²: {self.r2:.4f}
        - ¿Coeficientes y p-values: {coef_pval} y responde estrictamente en este orden y formato:

        - Variables Significativas: Lista solo las variables con p-value < 0.05 (es decir, cuya relación con las ventas es estadísticamente confiable). Ej: - Precio (p-value: 0.01).

        - Impacto Principal: Identifica la una variable del modelo con el coeficiente más alto (en valor absoluto). Traduce su impacto a un lenguaje claro:

        Ejemplo para Precio: - La variable con mayor impacto es el Precio. Por cada 1% que aumentes el precio, las ventas caerán aproximadamente un [Valor Absoluto del Coeficiente]%.

        - Calidad del Modelo (R²): - El modelo explica aproximadamente un [R²*100]% de las variaciones en la demanda. [Interpretación breve: ej. "Ajuste sólido" si R² > 0.7, "Ajuste moderado" si R² > 0.5, etc.].

        - Implicación Estratégica - Precio: Da una recomendación concreta y breve basada en la elasticidad-precio.

        Si es elástica (|coef. precio| > 1): - Estrategia de precio: Cuidado con las alzas. Una subida de precio generará una caída *más que proporcional* en la cantidad demandada, reduciendo los ingresos totales.

        Si es inelástica (|coef. precio| < 1): - Estrategia de precio: Oportunidad de margen. Puedes aumentar el precio; la caída en ventas será *menos que proporcional*, aumentando los ingresos totales.
        
        - Implicación Estratégica - Otras Variables (ej. Clima, PRECIO_COMPETENCIA): Si hay variables significativas además del precio, elige la más importante y da una recomendación.

        Ejemplo para Temperatura: - Acción Comercial: En días con mayor temperatura, espera un aumento de ~[e^(coeficiente)]% en ventas. Asegura stock y visibilidad en tienda.

        Si faltan datos para la Tarea 1, omítela y comienza directamente con la Tarea 2."""
                

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=st.secrets["HUGGINGFACE"]["HF_TOKEN_2"],
        )

        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": template}],
            temperature=0.3,
        )

        resultado = completion.choices[0].message.content.strip()
        print(resultado)
        return resultado

    

