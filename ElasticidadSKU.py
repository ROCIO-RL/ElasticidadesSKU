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
import plotly.express as px
import streamlit as st
from openai import OpenAI, APIError, APIStatusError



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
query_int =f"""
            SELECT 
                Pais,
                TMPANIOSEMANAGENOMMA, 
                TMPSEMANAANIOGENOMMA, 
                TEMPMAX 
            FROM PRD_CNS_MX.SO_HECHOS.VW_DATOS_CLIMA_INT 
            WHERE TMPANIOSEMANAGENOMMA>=2023
            """

query2="""SELECT 
            PROPSTCODBARRAS,
            MIN(PROPSTID) AS PROPSTID 
        FROM PRD_CNS_MX.DM.VW_DIM_PRODUCTO 
        GROUP BY PROPSTCODBARRAS"""
clima_bd = pd.read_sql(query,conn)
clima_bd.columns=['Año','Sem','Temperatura']
clima_bd_int = pd.read_sql(query_int,conn)
clima_bd_int.columns=['Pais','Año','Sem','Temperatura']
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
    def __init__(self, codbarras, canal, temp,grps,productobase,desc_competencias,pais,ruta_competencia="Competencia_Elasticidades.xlsx"):
        """
        codbarras: Código de barras del producto
        canal: 'Autoservicios', 'Farmacias' o 'Moderno'
        temp: True/False si se desea incluir clima
        """
        self.pais = pais
        self.codbarras = codbarras
        self.canal = canal
        self.temp = temp
        self.grps = grps
        self.producto_base =productobase
        self.ruta_competencia = ruta_competencia
        #self.precio_competencia = None 
        #self.nombre_competencia = desc_competencia
        self.precio_competencia = {}
        self.nombre_competencias = desc_competencias if isinstance(desc_competencias, list) else [desc_competencias]
        self.ultima_Semana = None
        self.status= None
        self.grps_actuales = 0
        self.columnas = None

    '''def calcula_precio(self, venta):
        if self.pais=='México':
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
        if self.pais=='México':
            # Aplicar IVA
            tasa_iva = 0.16
            venta['Precio'] = venta['Precio'] * (1 + tasa_iva)

        # Promedio semanal
        if self.canal in ['Autoservicios','Moderno']:
            clientes_por_semana = venta.groupby(['ANIO','SEMNUMERO'])['CADID'].nunique().reset_index()
            semanas_validas = clientes_por_semana[clientes_por_semana['CADID'] >= 3][['ANIO','SEMNUMERO']]
            venta = venta.merge(semanas_validas, on=['ANIO','SEMNUMERO'])
        
        # --- Conversión a USD si el país es Argentina ---
        if self.pais == 'Argentina':
            preciosarg = pd.read_excel(r"PrecioArg.xlsx",sheet_name='PreciosArg')
            preciosarg = preciosarg[preciosarg['PROPSTCODBARRAS']==self.codbarras]
            conn = snowflake.connector.connect(
                user=st.secrets["snowflake"]["user"],
                password=st.secrets["snowflake"]["password"],
                account=st.secrets["snowflake"]["account"],
                database=st.secrets["snowflake"]["database"],
                schema=st.secrets["snowflake"]["schema"]
            )

            query = """
            SELECT 
                t.TMPANIOSEMANAGENOMMA AS ANIO,
                t.TMPSEMANAANIOGENOMMA AS SEMNUMERO,
                AVG(USD) AS ML_USD
            FROM PRD_CNS_MX.CATALOGOS.VW_TIPO_CAMBIO_DIARIO AS d
            LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_CAT_TIEMPO AS t 
                ON t.TMPID = d.TMPID
            WHERE d.PAISID = 9
            AND t.TMPID >= 20230101
            GROUP BY 
                t.TMPANIOSEMANAGENOMMA, 
                t.TMPSEMANAANIOGENOMMA
            ORDER BY 
                t.TMPANIOSEMANAGENOMMA, 
                t.TMPSEMANAANIOGENOMMA
            """
            dolares = pd.read_sql(query, conn)
            conn.close()

            dolares['ANIO'] = dolares['ANIO'].astype(int)
            dolares['SEMNUMERO'] = dolares['SEMNUMERO'].astype(int)

            # Merge directo con tus semanas
            venta = venta.merge(dolares, on=['ANIO', 'SEMNUMERO'], how='left')

            # Conversión a USD
            venta['Precio'] = venta['Precio'] / venta['ML_USD']

        precio = venta.groupby(['ANIO','SEMNUMERO'])['Precio'].mean().reset_index()

        return precio
'''



    
    def preparar_grps(self):
       
        

        # medios
        data_medios = pd.read_csv(r"DashboardInternacional_ProductoBase.csv")

    
        # Filtrar
        data_medios = data_medios[data_medios['Pais'] == self.pais]
        data_medios = data_medios[data_medios['Producto base'] == self.producto_base ]

        if data_medios.empty:
            print(f"No se encontraron GRPs para '{self.producto_base }' ({self.pais}).")
            self.grps = False
            return pd.DataFrame()

        # Agrupar
        data_medios = (
            data_medios.groupby(['Año', 'Sem'], as_index=False)['Grps'].sum()
            .rename(columns={'Año': 'ANIO', 'Sem': 'SEMNUMERO'})
        )
        self.status = 3

        print(f"{len(data_medios)} filas de GRPs encontradas para {self.producto_base }.")
        return data_medios




    #funcion sustituida
    def calcula_precio(self, venta):

        # --- CASO MEXICO ---
        if self.pais == 'México':
            # Filtrado según canal
            if self.canal == 'Autoservicios':
                venta = venta[
                    (venta['CADID'].isin([2, 1, 15, 18, 3, 593])) &
                    (venta['MONTORETAIL'] > 0) &
                    (venta['UNIDADESDESP'] > 0)
                ].copy()
            elif self.canal == 'Farmacias':
                venta = venta[
                    (venta['CADID'].isin([27, 29])) &
                    (venta['MONTORETAIL'] > 0) &
                    (venta['UNIDADESDESP'] > 0)
                ].copy()
            elif self.canal == 'Moderno':
                venta = venta[
                    (venta['CADID'].isin([1, 27, 18, 15, 2, 16, 3, 593])) &
                    (venta['MONTORETAIL'] > 0) &
                    (venta['UNIDADESDESP'] > 0)
                ].copy()

            # Precio unitario
            venta['Precio'] = venta['MONTORETAIL'] / venta['UNIDADESDESP']

            # Aplicar IVA
            tasa_iva = 0.16
            venta['Precio'] = venta['Precio'] * (1 + tasa_iva)

            # Filtrar semanas válidas
            if self.canal in ['Autoservicios', 'Moderno']:
                clientes_por_semana = (
                    venta.groupby(['ANIO', 'SEMNUMERO'])['CADID']
                    .nunique()
                    .reset_index()
                )
                semanas_validas = clientes_por_semana[
                    clientes_por_semana['CADID'] >= 3
                ][['ANIO', 'SEMNUMERO']]
                venta = venta.merge(semanas_validas, on=['ANIO', 'SEMNUMERO'])

        # --- CASO ARGENTINA ---
        elif self.pais == 'Argentina':
            # Leemos precios externos
            preciosarg = pd.read_excel(r"PrecioArg.xlsx", sheet_name='PreciosArg')
            preciosarg.columns = [c.strip() for c in preciosarg.columns]
            preciosarg = preciosarg.rename(columns={
                'Promedio de Precio': 'Precio',
                'SEMANA': 'SEMNUMERO' if 'SEMANA' in preciosarg.columns else 'SEMNUMERO'
            })
            preciosarg = preciosarg[
                preciosarg['PROPSTCODBARRAS'] == self.codbarras
            ][['ANIO', 'SEMNUMERO', 'Precio']]

            # --- Merge con venta ---
            venta = venta.merge(preciosarg, on=['ANIO', 'SEMNUMERO'], how='left')

            # --- Obtener tipo de cambio semanal ---
            conn = snowflake.connector.connect(
                user=st.secrets["snowflake"]["user"],
                password=st.secrets["snowflake"]["password"],
                account=st.secrets["snowflake"]["account"],
                database=st.secrets["snowflake"]["database"],
                schema=st.secrets["snowflake"]["schema"]
            )

            query = """
            SELECT 
                t.TMPANIOSEMANAGENOMMA AS ANIO,
                t.TMPSEMANAANIOGENOMMA AS SEMNUMERO,
                AVG(USD) AS ML_USD
            FROM PRD_CNS_MX.CATALOGOS.VW_TIPO_CAMBIO_DIARIO AS d
            LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_CAT_TIEMPO AS t 
                ON t.TMPID = d.TMPID
            WHERE d.PAISID = 9
            AND t.TMPID >= 20230101
            GROUP BY 
                t.TMPANIOSEMANAGENOMMA, 
                t.TMPSEMANAANIOGENOMMA
            ORDER BY 
                t.TMPANIOSEMANAGENOMMA, 
                t.TMPSEMANAANIOGENOMMA
            """
            dolares = pd.read_sql(query, conn)
            conn.close()

            dolares[['ANIO', 'SEMNUMERO']] = dolares[['ANIO', 'SEMNUMERO']].astype(int)

            # --- Merge con tipo de cambio ---
            venta = venta.merge(dolares, on=['ANIO', 'SEMNUMERO'], how='left')

            # --- Interpolar precios faltantes ---
            venta = venta.sort_values(['ANIO', 'SEMNUMERO'])
            venta['Precio'] = venta['Precio'].interpolate(method='linear', limit_direction='both')

            # --- Conversión a USD ---
            venta['Precio'] = venta['Precio'] / venta['ML_USD']

            # --- Quitar atípicos usando IQR ---
            Q1 = venta['Precio'].quantile(0.25)
            Q3 = venta['Precio'].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR

            # Filtrar valores dentro del rango válido
            venta = venta[
                (venta['Precio'] >= limite_inferior) &
                (venta['Precio'] <= limite_superior)
            ].copy()

            # --- Promedio semanal final ---
            precio = (
                venta.groupby(['ANIO', 'SEMNUMERO'])['Precio']
                .mean()
                .reset_index()
            )

            return precio


        else:
            #print(f"País {self.pais} no contemplado.")
            #return pd.DataFrame()
             # Precio unitario
            #venta['Precio'] = venta['MONTORETAIL'] / venta['UNIDADESDESP']
            venta['Precio'] = venta['MONTORETAIL'] / venta['UNIDADESDESP'].replace(0, np.nan)
            

            

        # --- Promedio semanal final ---
        precio = (
            venta.groupby(['ANIO', 'SEMNUMERO'])['Precio']
            .mean()
            .reset_index()
        )
        #return
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

        query_int = f"""SELECT 
            s.SEMANIO AS ANIO, 
            s.SEMNUMERO AS SEMNUMERO,
            es.PROPSTCODBARRAS,
            so.cadid,
            so.SOUTCANTDESP AS UnidadesDesp,
            NVL(so.soutmontodesp,0) AS MontoRetail
        FROM PRD_CNS_MX.DM.FACT_SO_SEM_CAD_SKU_INT so 
        LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_ESTRUCTURAPRODUCTOSTOTALPAISES es ON es.PROPSTID=so.PROPSTID 
        LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_ESTRUCTURACLIENTESSEGPTVTOTAL cl ON cl.CADID=so.CADID  
        LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_CATSEMANAS s ON s.SEMID=so.SEMID 
        LEFT JOIN PRD_STG.GNM_CT.GNMPAIS p ON p.PAISID=so.PAISID  
        WHERE s.SEMANIO>=2023   
                AND P.PAIS='{self.pais}'
                AND es.propstcodbarras = '{self.codbarras}'
                AND cl.TIPOESTNOMBRE IN ('Autoservicios')
                AND cl.GRPCLASIFICACION='Monitoreado' 
        """   

        query_int_2 = f"""SELECT 
            s.SEMANIO AS ANIO, 
            s.SEMNUMERO AS SEMNUMERO,
            es.PROPSTCODBARRAS,
            so.cadid,
            so.SOUTCANTDESP AS UnidadesDesp,
            NVL(so.soutmontodespbrt,0) AS MontoRetail
        FROM PRD_CNS_MX.DM.FACT_SO_SEM_CAD_SKU_INT so 
        LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_ESTRUCTURAPRODUCTOSTOTALPAISES es ON es.PROPSTID=so.PROPSTID 
        LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_ESTRUCTURACLIENTESSEGPTVTOTAL cl ON cl.CADID=so.CADID  
        LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_CATSEMANAS s ON s.SEMID=so.SEMID 
        LEFT JOIN PRD_STG.GNM_CT.GNMPAIS p ON p.PAISID=so.PAISID  
        WHERE s.SEMANIO>=2023   
                AND P.PAIS='{self.pais}'
                AND es.propstcodbarras = '{self.codbarras}'
                AND cl.TIPOESTNOMBRE IN ('Autoservicios')
                AND cl.GRPCLASIFICACION='Monitoreado' 
        """  

        if self.pais=='México':
            self.sellout = pd.read_sql(query, conn)
            conn.close()
            # Filtro de cadenas según canal
            if self.canal == 'Autoservicios':
                self.sellout = self.sellout[self.sellout['CADID'].isin([1,10,100,102,15,16,18,19,2,20,21,25,3,342,380,4,5,593,652,11,12,13,381,493,6,9])]
            elif self.canal == 'Farmacias':
                self.sellout = self.sellout[~self.sellout['CADID'].isin([1,10,100,102,15,16,18,19,2,20,21,25,3,342,380,4,5,593,652,11,12,13,381,493,6,9])]

        else:
            if self.pais =='Colombia':
                self.sellout = pd.read_sql(query_int_2, conn)
            else:
                self.sellout = pd.read_sql(query_int, conn)
            #conn.close()

            # Verificar si todo el periodo carece de neto (es nulo o cero)
            '''if self.sellout['MONTODESPNETO'].isnull().all() or (self.sellout['MONTODESPNETO'] == 0).all():
                # Todo el periodo sin neto → usar bruto
                self.sellout['MONTORETAIL'] = self.sellout['MONTODESPBRUTO']
            else:
                # Si hay al menos un neto válido → aplicar regla fila por fila
                self.sellout['MONTORETAIL'] = self.sellout['MONTODESPNETO'].where(
                    self.sellout['MONTODESPNETO'].notnull() & (self.sellout['MONTODESPNETO'] != 0),
                    self.sellout['MONTODESPBRUTO']
                )'''
            

            query_cadid = f"""
                SELECT DISTINCT
                    P.PAIS,
                    so.cadid,
                    cl.TIPOESTNOMBRE
                FROM PRD_CNS_MX.DM.FACT_SO_SEM_CAD_SKU_INT so 
                LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_ESTRUCTURAPRODUCTOSTOTALPAISES es ON es.PROPSTID=so.PROPSTID 
                LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_ESTRUCTURACLIENTESSEGPTVTOTAL cl ON cl.CADID=so.CADID  
                LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_CATSEMANAS s ON s.SEMID=so.SEMID 
                LEFT JOIN PRD_STG.GNM_CT.GNMPAIS p ON p.PAISID=so.PAISID  
                WHERE s.SEMANIO>=2023   
                AND cl.TIPOESTNOMBRE IN ('Autoservicios','Cadenas de farmacia')
                AND cl.GRPCLASIFICACION='Monitoreado'
                AND P.PAIS='{self.pais}'
                """

            df_cadid = pd.read_sql(query_cadid, conn)
            conn.close()
            cadid_autoserv = df_cadid.loc[df_cadid['TIPOESTNOMBRE'] == 'Autoservicios', 'CADID'].unique().tolist()
            cadid_farm = df_cadid.loc[df_cadid['TIPOESTNOMBRE'] == 'Cadenas de farmacia', 'CADID'].unique().tolist()
            # Filtro de cadenas según canal
            if self.canal == 'Autoservicios':
                self.sellout = self.sellout[self.sellout['CADID'].isin(cadid_autoserv)]
            elif self.canal == 'Farmacias':
                self.sellout = self.sellout[~self.sellout['CADID'].isin(cadid_farm)]





        
        # Precios semanales
        self.precio_gli = self.calcula_precio(self.sellout)

        # Ventas semanales + precio promedio
        self.sellout = self.sellout.groupby(['ANIO','SEMNUMERO']).agg({'UNIDADESDESP':'sum'}).reset_index()
        self.sellout = self.sellout.merge(self.precio_gli, on=['ANIO','SEMNUMERO'], how='left')

        return self.sellout
    
        
    '''def carga_competencia(self):
        try:
            comp = pd.read_excel(r"Competencias_Elasticidades.xlsx")
            comp.columns = [c.strip() for c in comp.columns]
            comp = comp.rename(columns={
                'SKU': 'PROPSTCODBARRAS',
                'Descripcion Competencia': 'DESC_COMPETENCIA',
                'Precio Competencia': 'PRECIO_COMPETENCIA'
            })
            comp = comp[['PROPSTCODBARRAS','ANIO','DESC_COMPETENCIA','SEMNUMERO','PRECIO_COMPETENCIA']]
            comp['PROPSTCODBARRAS'] = comp['PROPSTCODBARRAS'].astype(str).str.strip()
            comp = comp[comp['PROPSTCODBARRAS'] == self.codbarras]  # filtrar por el SKU actual
           
            #primera_desc = comp['DESC_COMPETENCIA'].iloc[1]
            # Filtrar todas las filas que tengan esa misma descripción
            #comp = comp[comp['DESC_COMPETENCIA'] == primera_desc]
            if not comp.empty:
                # Tomar la primera descripción disponible
                #primera_desc = comp['DESC_COMPETENCIA'].iloc[0]
                #self.nombre_competencia = primera_desc
                # Filtrar todas las filas que tengan esa descripción
                #comp = comp[comp['DESC_COMPETENCIA'] == self.nombre_competencia]
                #comp = comp[['PROPSTCODBARRAS','ANIO','SEMNUMERO','PRECIO_COMPETENCIA']]


                # Si no hay nombre de competencia o no se encuentra en los datos
                if not self.nombre_competencia or self.nombre_competencia not in comp['DESC_COMPETENCIA'].values:
                    # Tomar la primera descripción disponible
                    self.nombre_competencia = comp['DESC_COMPETENCIA'].iloc[0]
                
                # Filtrar todas las filas que tengan esa descripción
                comp = comp[comp['DESC_COMPETENCIA'] == self.nombre_competencia]
                comp = comp[['PROPSTCODBARRAS','ANIO','SEMNUMERO','PRECIO_COMPETENCIA']]
            print(comp)
            return comp
        except Exception as e:
            print(f"No se pudo cargar competencia: {e}")
            return pd.DataFrame()'''
    def carga_competencia(self):
        try:
            self.status=10
            comp = pd.read_excel(r"Competencias_Elasticidades_VF.xlsx")
            comp = comp[comp['Pais']==self.pais]
            comp.columns = [c.strip() for c in comp.columns]
            comp = comp.rename(columns={
                'SKU': 'PROPSTCODBARRAS',
                'Descripcion Competencia': 'DESC_COMPETENCIA',
                'Precio Competencia': 'PRECIO_COMPETENCIA'
            })
            comp = comp[['PROPSTCODBARRAS','ANIO','DESC_COMPETENCIA','SEMNUMERO','PRECIO_COMPETENCIA']]
            comp['PROPSTCODBARRAS'] = comp['PROPSTCODBARRAS'].astype(str).str.strip()
            comp = comp[comp['PROPSTCODBARRAS'] == self.codbarras]

            # Filtramos solo las competencias seleccionadas
            comp = comp[comp['DESC_COMPETENCIA'].isin(self.nombre_competencias)]

            if comp.empty:
                print("No se encontró información de competencia.")
                self.status=0
                return pd.DataFrame()

            
            # --- Conversión a USD si el país es Argentina ---
            '''if self.pais == 'Argentina':
                conn = snowflake.connector.connect(
                    user=st.secrets["snowflake"]["user"],
                    password=st.secrets["snowflake"]["password"],
                    account=st.secrets["snowflake"]["account"],
                    database=st.secrets["snowflake"]["database"],
                    schema=st.secrets["snowflake"]["schema"]
                )

                query = """
                SELECT 
                    t.TMPANIOSEMANAGENOMMA AS ANIO,
                    t.TMPSEMANAANIOGENOMMA AS SEMNUMERO,
                    AVG(USD) AS ML_USD
                FROM PRD_CNS_MX.CATALOGOS.VW_TIPO_CAMBIO_DIARIO AS d
                LEFT JOIN PRD_CNS_MX.CATALOGOS.VW_CAT_TIEMPO AS t 
                    ON t.TMPID = d.TMPID
                WHERE d.PAISID = 9
                AND t.TMPID >= 20230101
                GROUP BY 
                    t.TMPANIOSEMANAGENOMMA, 
                    t.TMPSEMANAANIOGENOMMA
                ORDER BY 
                    t.TMPANIOSEMANAGENOMMA, 
                    t.TMPSEMANAANIOGENOMMA
                """
                dolares = pd.read_sql(query, conn)
                conn.close()

                dolares['ANIO'] = dolares['ANIO'].astype(int)
                dolares['SEMNUMERO'] = dolares['SEMNUMERO'].astype(int)

                # Merge con los precios de competencia
                comp = comp.merge(dolares, on=['ANIO', 'SEMNUMERO'], how='left')

                # Convertir precios a USD
                comp['PRECIO_COMPETENCIA'] = comp['PRECIO_COMPETENCIA'] / comp['ML_USD']'''


            #cambios
            self.status=1
            # Pivot para tener una columna por competencia
            comp_pivot = comp.pivot_table(
                index=['ANIO', 'SEMNUMERO'],
                columns='DESC_COMPETENCIA',
                values='PRECIO_COMPETENCIA'
            ).reset_index()

            # Renombramos columnas para evitar espacios
            comp_pivot.columns = [f"PRECIO_COMPETENCIA_{str(c).replace(' ', '_')}" if c not in ['ANIO','SEMNUMERO'] else c for c in comp_pivot.columns]


            import re

            def limpiar_nombre_comp(nombre):
                # Reemplazar caracteres inválidos por _
                return re.sub(r'[^A-Za-z0-9_]', '_', str(nombre))

            comp_pivot.columns = [
                f"PRECIO_COMPETENCIA_{limpiar_nombre_comp(c)}" if c not in ['ANIO', 'SEMNUMERO'] else c
                for c in comp_pivot.columns
            ]


            # Guardamos últimos precios por competencia
            for col in comp_pivot.columns:
                if col.startswith('PRECIO_COMPETENCIA'):
                    self.precio_competencia[col] = comp_pivot[col].dropna().iloc[-1] if comp_pivot[col].notna().any() else None

            return comp_pivot

        except Exception as e:
            print(f"No se pudo cargar competencia: {e}")
            return pd.DataFrame()



    def prepara_datos(self):
        layout = self.sellout[(self.sellout['UNIDADESDESP'] > 0) & (self.sellout['Precio'].notna())].copy()
        layout = layout[layout['ANIO'] >= 2023]

        #print(f"Valores nulos:\n{layout.isna().sum()}\nBorrando nulos...")

        if layout['Precio'].isna().sum() > 30:
            raise ValueError("El dataframe contiene demasiados nulos")

        layout.dropna(inplace=True)


        if self.pais=='México':
            # Dummy de Julio Regalado
            layout["JULIO_REGALADO"] = np.where(layout["SEMNUMERO"].between(21, 31), 1, 0)
            print("Variable dummy 'JULIO_REGALADO' agregada correctamente.")

            # Dummy de Mega Pauta
            layout["MEGA_PAUTA"] = np.where(layout["SEMNUMERO"].between(1, 6), 1, 0)
            print("Variable dummy 'MEGA_PAUTA' agregada correctamente.")

            #clima
            if self.temp:
                temperatura = clima_bd.copy()
                temperatura.columns = ['ANIO','SEMNUMERO','CLIMA']
                layout = layout.merge(temperatura, on=['ANIO','SEMNUMERO'], how='left')
        else:
            #clima
            if self.temp:
                temperatura = clima_bd_int.copy()
                temperatura = temperatura[temperatura['Pais'] == self.pais]
                temperatura.columns = ['Pais','ANIO','SEMNUMERO','CLIMA']
                temperatura = temperatura[['ANIO','SEMNUMERO','CLIMA']]
                layout = layout.merge(temperatura, on=['ANIO','SEMNUMERO'], how='left')
        try:
            data_grps = self.preparar_grps()
            if self.grps:
                #data_grps = self.preparar_grps()
                layout = layout.merge(data_grps, on=['ANIO', 'SEMNUMERO'], how='left')
                layout["Grps"] = layout["Grps"].fillna(0)
                self.grps_actuales = layout['Grps'].iloc[-1]

        except Exception as e:
            print(f"No se pudo cargar competencia: {e}")

           
                
        
        self.columnas = layout.columns

               
                




        # Competencia
        '''competencia = self.carga_competencia()
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
            print("No se encontró información de competencia para este SKU.")'''
        # Competencia
        competencias = self.carga_competencia()
        if not competencias.empty:
            layout = layout.merge(competencias, on=['ANIO','SEMNUMERO'], how='left')
            print("Información de competencia agregada correctamente.")
        else:
            print("No se encontró información de competencia para este SKU.")


        layout_log = layout.copy()
        self.data_grafico = layout.copy()

        # log-log
        layout_log[['UNIDADESDESP','Precio']] = layout_log[['UNIDADESDESP','Precio']].apply(np.log)
        # competencia
        '''if 'PRECIO_COMPETENCIA' in layout_log.columns and layout_log['PRECIO_COMPETENCIA'].notna().sum() > 0:
            layout_log['PRECIO_COMPETENCIA'] = np.log(layout_log['PRECIO_COMPETENCIA'])'''
        # log-log de precios de competencia
        for col in layout_log.columns:
            if col.startswith('PRECIO_COMPETENCIA'):
                layout_log[col] = np.log(layout_log[col])


        # Guardar última semana y año
        if not layout.empty:
            ultimo_anio = layout['ANIO'].max()
            ultima_semana = layout[layout['ANIO'] == ultimo_anio]['SEMNUMERO'].max()
            self.ultima_semana = (int(ultimo_anio), int(ultima_semana))
            print(f"Última semana registrada: Año {ultimo_anio}, Semana {ultima_semana}")
        else:
            self.ultima_semana = None

        
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

        '''if self.temp:
            formula += ' + CLIMA'

        if 'PRECIO_COMPETENCIA' in data.columns and data['PRECIO_COMPETENCIA'].notna().sum() > 0:
            formula += ' + PRECIO_COMPETENCIA'
            '''
        
        if self.temp:
            formula += ' + CLIMA'

        #if self.grps:
        #    formula += ' + Grps'
        if self.grps and 'Grps' in data.columns and not data['Grps'].isna().all():
            formula += ' + Grps'
        '''if 'Grps' in data.columns:
            print("Resumen de Grps:", data['Grps'].describe())
            if data['Grps'].dropna().empty:
                print("⚠️ Grps está vacío. Se removerá de la fórmula.")
                data = data.drop(columns=['Grps'], errors='ignore')
                self.grps = False'''

        # Agregar todas las competencias
        for col in data.columns:
            if col.startswith('PRECIO_COMPETENCIA'):
                formula += f' + {col}'


        if self.pais=='México':
            # Agregamos la dummy de Julio Regalado
            if 'JULIO_REGALADO' in data.columns:
                formula += ' + JULIO_REGALADO'

            # Agregamos la dummy de Julio Regalado
            if 'MEGA_PAUTA' in data.columns:
                formula += ' + MEGA_PAUTA'

        print(f"Fórmula del modelo: {formula}")
        modelo = smf.ols(formula, data=data).fit()

        print(modelo.summary())
        self.r2 = modelo.rsquared
        self.coeficientes = modelo.params
        self.pvalores = modelo.pvalues
    


    '''def grafica(self):
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

        return fig'''
    
    def grafica(self):
        df = self.data_grafico.copy()

        # Crear columna de fecha
        df['Fecha'] = pd.to_datetime(
            df['ANIO'].astype(str) +
            df['SEMNUMERO'].astype(str).str.zfill(2) + '1', 
            format='%G%V%u'
        )

        # Crear figura
        fig = go.Figure()

        # Ventas (barras)
        fig.add_trace(go.Bar(
            x=df['Fecha'],
            y=df['UNIDADESDESP'],
            name='Unidades Vendidas',
            marker_color='lightgray',
            yaxis='y1'
        ))

        # --- 2️⃣ Precio propio ---
        fig.add_trace(go.Scatter(
            x=df['Fecha'],
            y=df['Precio'],
            name='Precio Propio',
            mode='lines',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))

        # Competencias
        # Buscar todas las columnas que empiecen con PRECIO_COMPETENCIA
        cols_comp = [c for c in df.columns if c.startswith('PRECIO_COMPETENCIA')]

        if cols_comp:
            colores = px.colors.qualitative.Dark24  # paleta de 24 colores
            for i, col in enumerate(cols_comp):
                nombre_comp = col.replace('PRECIO_COMPETENCIA', '').replace('_', ' ')
                fig.add_trace(go.Scatter(
                    x=df['Fecha'],
                    y=df[col],
                    name=f'Competencia: {nombre_comp}',
                    mode='lines',
                    line=dict(color=colores[i % len(colores)], width=2, dash='dot'),
                    yaxis='y2'
                ))

        # Clima 
        '''if 'CLIMA' in df.columns:
            temp_norm = (df['CLIMA'] - df['CLIMA'].min()) / (df['CLIMA'].max() - df['CLIMA'].min())
            fig.add_trace(go.Scatter(
                x=df['Fecha'],
                y=temp_norm,
                fill='tozeroy',
                mode='none',
                name='Clima (normalizado)',
                fillcolor='rgba(0, 160, 220, 0.2)',
                yaxis='y3'
            ))'''
        
        if 'Grps' in df.columns:
            grps_norm = (df['Grps'] - df['Grps'].min()) / (df['Grps'].max() - df['Grps'].min())
            fig.add_trace(go.Scatter(
                x=df['Fecha'],
                y=grps_norm,
                fill='tozeroy',
                mode='none',
                name='Grps (normalizado)',
                fillcolor='rgba(0, 160, 220, 0.2)',
                yaxis='y3'
            ))
        else:
            # Clima 
            if 'CLIMA' in df.columns:
                temp_norm = (df['CLIMA'] - df['CLIMA'].min()) / (df['CLIMA'].max() - df['CLIMA'].min())
                fig.add_trace(go.Scatter(
                    x=df['Fecha'],
                    y=temp_norm,
                    fill='tozeroy',
                    mode='none',
                    name='Clima (normalizado)',
                    fillcolor='rgba(0, 160, 220, 0.2)',
                    yaxis='y3'
                ))

        #Layout
        #control
        fig.update_layout(
            title="Unidades vendidas vs Precio propio y Variables externas",
            xaxis=dict(title="Semana"),
            yaxis=dict(title="Unidades", side='left', showgrid=False),
            yaxis2=dict(title="Precio", overlaying='y', side='right'),
            yaxis3=dict(title="Variables Externas (escala visual)",
                        overlaying='y', side='right', position=0.95, showgrid=False),
            bargap=0.2,
            legend=dict(
                orientation='h',
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                title=None,
                bgcolor='rgba(255,255,255,0.5)'
            ),
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
                api_key=st.secrets["HUGGINGFACE"]["HF_TOKEN_3"], 
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

        template = f"""Contexto:
                    Eres un Econometrista Senior especializado en transformar resultados estadísticos complejos en recomendaciones de negocio claras, accionables y ejecutivas para equipos de dirección comercial.
                    Tu lenguaje es directo, ejecutivo y basado en evidencia cuantitativa. No usas jerga innecesaria, es decir pesos $.

                    ⚙️ Instrucciones de Respuesta

                    Formato:

                    Usa exclusivamente viñetas (°), sin introducción, sin conclusión.

                    Máximo 6 viñetas.

                    Lenguaje: español, con términos técnicos explicados brevemente entre paréntesis cuando aparezcan.

                    🧩 Datos que recibirás (variables del modelo)

                    R²: {self.r2:.4f}

                    coeficientes: {coef_pval}

                    precio: {precio}

                    df: {df} (tabla de demanda simulada)

                    Variables posibles: Precio, CLIMA, PRECIO_COMPETENCIA_xxx, JULIO_REGALADO, u otras.

                    🧾 Estructura esperada de salida
                    🔹 Tarea 1: Recomendación de Precio (solo si hay datos de demanda y precio actual)

                    Precio Ideal Propuesto: [valor estimado].

                    Rango Alternativo: [$X - $Y]. Ventaja: [p. ej. “mejora el margen sin pérdida significativa de volumen”].

                    🔹 Tarea 2: Interpretación del Modelo Log-Log

                    Variables Significativas: Lista solo las variables con p-value < 0.05 (es decir, relación estadísticamente confiable con las ventas).
                    Ejemplo: - Precio (p-value: 0.01).

                    Impacto Principal:
                    Identifica la variable con mayor impacto (coeficiente de mayor valor absoluto) y tradúcelo:
                    Ejemplo:
                    - El precio tiene el mayor impacto. Un aumento del 1% reduce las ventas en ~[|β|]%.

                    Calidad del Modelo (R²):
                    - El modelo explica ~[R²*100]% de la variación en las ventas. Clasificación: sólido (R²>0.7), moderado (0.5–0.7), débil (<0.5).

                    💰 Tarea 3: Estrategia de Precio según tipo de elasticidad

                    Si |β_precio| > 1 → Elástica:
                    - Estrategia de precio: Evita alzas. Un incremento de precio reduce las ventas de forma más que proporcional, afectando ingresos totales.

                    Si |β_precio| < 1 → Inelástica:
                    - Estrategia de precio: Oportunidad de margen. Puedes aumentar precios; la caída en volumen será menor al aumento en ingresos.

                    🤝 Tarea 4: Análisis de la Elasticidad Cruzada (Competencia)

                    Interpreta todas las variables PRECIO_COMPETENCIA_xxx bajo esta lógica:

                    β > 0 → Relación sustitutiva: si la competencia sube su precio, tus ventas aumentan.

                    β < 0 → Co-movimiento o complementariedad aparente: tus ventas se mueven junto con la competencia.

                    |β| > 1 → Efecto elástico; revisar si refleja factores externos.

                    Incluye viñetas como:

                    - Relación con Competencia [nombre]: Sustitutiva (coef. +0.45). Si la competencia sube 1%, tus ventas crecen ~0.45%.

                    - Relación con Competencia [nombre]: Co-movimiento aparente (coef. -1.2). Ambas marcas responden a factores externos como promociones simultáneas (ej. Julio Regalado).

                    🌤️ Tarea 5: Otras Variables Significativas

                    Si el modelo incluye CLIMA, JULIO_REGALADO, MEGA_PAUTA u otras:

                    - Clima: Por cada aumento del 1% en temperatura, las ventas cambian en ~[β_clima*100]%

                    - Julio Regalado: Incrementa ventas en ~[β_JR*100]% durante semanas 21–31.
                    
                    - Mega Pauta: Las ventas incrementan en ~[β_MP*100]% durante semanas 01–06.

                    ⚠️ Tarea 6: Consideraciones Analíticas y Recomendaciones Avanzadas

                    Incluye solo una o dos viñetas finales de alto nivel:

                    - Coeficiente cruzado negativo y elástico no implica sustitución; indica co-movimiento por choques comunes (ej. estacionalidad o promociones masivas).

                    - Se recomienda incorporar variables de control (estacionalidad, intensidad promocional o gasto publicitario) para aislar el efecto competitivo real.

                    - Añadir dummies de eventos de oferta y mega pauta (p. ej. Julio Regalado y Mega pauta) mejora la precisión del modelo."""
        # Anterior prompt
        """Contexto: Eres un Econometrista Senior que se especializa en transformar resultados estadísticos complejos en recomendaciones estratégicas claras y accionables para la alta dirección. Tu estilo es directo, ejecutivo y basado en datos.

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

         - Consideración Clave - Coeficientes Cruzados Atípicos: Si el coeficiente del precio de la competencia es negativo y elástico (|coef.| > 1), esto NO indica sustitución. Sugiere que ambas marcas se mueven juntas por factores externos (ej. promociones agregadas como "Julio Regalado"). Para aislar el efecto competitivo real, el modelo debe incluir variables de control como estacionalidad o pulsos de oferta.
        
        Si faltan datos para la Tarea 1, omítela y comienza directamente con la Tarea 2."""
                
        #HF_TOKEN_Apagado
        #status
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=st.secrets["HUGGINGFACE"]["HF_TOKEN_Apagado"],
        )


        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": template}],
                temperature=0.3,
            )
            resultado = completion.choices[0].message.content.strip()
            return resultado

        except (APIStatusError, APIError, Exception) as e:
            # Aquí atrapamos cualquier error de la API (por falta de créditos, timeouts, etc.)
            print(f"[ADVERTENCIA] Error generando insight con HuggingFace: {e}")
            return (
                "⚠️ **No se pudo generar el insight automático** debido a un error con el modelo "
                "(posiblemente créditos agotados o problema de conexión). "
                "Puedes continuar usando las gráficas sin inconveniente."
            )

    



