# LIBRERIAS
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
import os

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
    def __init__(self, codbarras, canal, temp):
        """
        codbarras: Código de barras del producto
        canal: 'Autoservicios', 'Farmacias' o 'Moderno'
        temp: True/False si se desea incluir clima
        """
        self.codbarras = codbarras
        self.canal = canal
        self.temp = temp

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

        layout_log = layout.copy()
        self.data_grafico = layout.copy()

        # log-log
        layout_log[['UNIDADESDESP','Precio']] = layout_log[['UNIDADESDESP','Precio']].apply(np.log)

        return layout_log

    def calcula_elasticidad(self):
        data = self.prepara_datos()

        if data.shape[0] < 30:
            raise ValueError("No hay datos suficientes para el modelo")

        if self.temp:
            modelo = smf.ols('UNIDADESDESP ~ Precio + CLIMA', data=data).fit()
        else:
            modelo = smf.ols('UNIDADESDESP ~ Precio', data=data).fit()

        print(modelo.summary())
        self.r2 = modelo.rsquared
        self.coeficientes = modelo.params
        self.pvalores = modelo.pvalues
    
    def grafica(self):
        self.data_grafico['Fecha'] = pd.to_datetime(
            self.data_grafico['ANIO'].astype(str) + 
            self.data_grafico['SEMNUMERO'].astype(str).str.zfill(2) + '1', format='%G%V%u'
        )
        
        fig, ax1 = plt.subplots(figsize=(14,6))
        
        # Barras de unidades
        ax1.bar(self.data_grafico['Fecha'], self.data_grafico['UNIDADESDESP'], 
                color='lightgray', label='Unidades', width=5.0)
        ax1.set_xlabel('Semana')
        ax1.set_ylabel('Unidades', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')
        
        # Línea de precio
        ax2 = ax1.twinx()
        ax2.plot(self.data_grafico['Fecha'], self.data_grafico['Precio'], 
                color='black', label='Precio GLI')
        precio_min = self.data_grafico['Precio'].min()
        precio_max = self.data_grafico['Precio'].max()
        rango = precio_max - precio_min
        margen_inf = max(precio_min - 0.4*rango, 0)
        margen_sup = precio_max + 0.1*rango
        ax2.set_ylim(margen_inf, margen_sup)
        
        # Línea de clima si existe
        if 'CLIMA' in self.data_grafico.columns:
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 60))  
            temp_norm = (self.data_grafico['CLIMA'] - self.data_grafico['CLIMA'].min()) / \
                        (self.data_grafico['CLIMA'].max() - self.data_grafico['CLIMA'].min())
            ax3.fill_between(self.data_grafico['Fecha'], temp_norm, 
                            alpha=0.25, color='lightcoral', label='CLIMA (normalizada)')
            ax3.set_ylabel('Clima (escala visual)', color='red')
            ax3.tick_params(axis='y', labelcolor='red')
        
        plt.title('Unidades vendidas vs Precio propio y clima')
        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.90))
        plt.xticks(rotation=45)
        
        return fig 

    
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
            - Coeficientes y p-values:
            {coef_pval}

            Tu tarea:
            - Responde en español.
            - Sé ejecutivo y breve.
            - Usa viñetas claras.
            - Da explicaciones pero, no seas extenso y enfócate en conclusiones de elasticidad. Tus respuestas deben dar valor al negocio.

            Incluye:
            1. Variables significativas.
            2. Variable con mayor impacto.
            3. Calidad del ajuste (R²).
            4. Implicaciones estratégicas de precios y clima.
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
    

