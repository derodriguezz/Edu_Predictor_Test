import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os

# Cargar archivo comparativo_historico.py automáticamente
file_path = r'./comparativo_historico.py'

# Función para cargar el archivo y las variables
def cargar_datos():
    # Importar el archivo comparativo_historico.py y cargar las variables
    with open(file_path) as f:
        exec(f.read(), globals())  # Ejecuta el código y carga las variables definidas en ese archivo

# Cargar datos automáticamente al iniciar la app
cargar_datos()

# Menú lateral
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=['Inicio', 'Pruebas', 'Implementación', 'Generar Indicativo', 'Realizar Predicción']
    )

# Sección de Inicio
if selected == 'Inicio':
    st.title(f"Herramientas de Modelos de Machine Learning")
    
    st.write("""
        Esta aplicación ofrece diversas herramientas para el análisis de datos históricos y la predicción de matrículas e inscritos 
        mediante el uso de modelos de machine learning. A través de las distintas secciones, podrás explorar los datos, visualizar
        los resultados de modelos y generar predicciones personalizadas.
        """)
    
    st.write("""
        - **Pruebas**: Explora el rendimiento de los modelos con los datos históricos.
        - **Implementación**: Conoce los resultados de las predicciones basadas en los modelos entrenados.
        - **Generar Indicativo**: Genera el indicativo anual de grupos por centro de formación (en desarrollo).
        - **Realizar Predicción**: Ejecuta una predicción personalizada para las inscripciones y matrículas (en desarrollo).
    """)

# Sección de Pruebas
if selected == 'Pruebas':
    st.title(f"Conozca el Rendimiento de los Modelos")
    
    st.write("""
        En esta sección, podrás explorar los datos históricos y conocer el rendimiento de los modelos que hemos implementado.
        Dispones de dos vistas: una con las tablas filtradas por años y otra con las visualizaciones (en construcción).
        """)
    
    # Pestañas para dividir en Tablas y Visualizaciones
    tab1, tab2 = st.tabs(["Tablas", "Visualizaciones"])

    # Pestaña de Tablas
    with tab1:
        st.subheader('Datos cargados - Comparativo Histórico')
        st.write("""
            Aquí puedes visualizar los datos históricos y filtrar por años específicos. Además, puedes agrupar los datos por 
            centro de formación para obtener un análisis más detallado.
        """)
        
        if 'comparativo_historico_mg' in globals():
            # Obtener los valores únicos de AÑO_FICHA
            años = comparativo_historico_mg['AÑO_FICHA'].unique()

            # Selector múltiple para los valores de AÑO_FICHA (primer tabla)
            años_seleccionados = st.multiselect("Seleccionar Año Ficha - Comparativo Histórico", años, default=años)

            # Filtrar el DataFrame según los años seleccionados
            df_filtrado = comparativo_historico_mg[comparativo_historico_mg['AÑO_FICHA'].isin(años_seleccionados)]

            # Mostrar la tabla filtrada
            st.dataframe(df_filtrado)

            st.subheader('Generar Tabla Agrupada por CODIGO_CENTRO')
            st.write("""
                Puedes seleccionar uno o varios años para agrupar los datos por centro de formación, sumando las matriculaciones
                reales y predichas.
            """)

            # Selector múltiple para los valores de AÑO_FICHA (segunda tabla)
            años_seleccionados_agrupado = st.multiselect("Seleccionar Año Ficha - Agrupado por Centro", años, default=años)

            # Filtrar el DataFrame según los años seleccionados para el agrupado
            df_filtrado_agrupado = comparativo_historico_mg[comparativo_historico_mg['AÑO_FICHA'].isin(años_seleccionados_agrupado)]

            # Agrupar por CODIGO_CENTRO y sumar MATRICULADOS y MATRICULADOS_PREDICHOS_19_24
            df_agrupado = df_filtrado_agrupado.groupby('CODIGO_CENTRO').agg(
                suma_matriculados=('MATRICULADOS', 'sum'),
                suma_matriculados_predichos=('MATRICULADOS_PREDICHOS_19_24', 'sum')
            ).reset_index()

            # Calcular la diferencia absoluta
            df_agrupado['diferencia_absoluta'] = abs(df_agrupado['suma_matriculados'] - df_agrupado['suma_matriculados_predichos'])

            # Calcular la proporción absoluta basada en la suma de MATRICULADOS
            df_agrupado['proporcion'] = df_agrupado['diferencia_absoluta'] / df_agrupado['suma_matriculados']

            # Mostrar la tabla agrupada
            st.dataframe(df_agrupado)

    # Pestaña de Visualizaciones
    with tab2:
        st.subheader('Visualizaciones - En Construcción')
        st.write("""
            En esta pestaña podrás ver gráficos y análisis visuales basados en los datos históricos. Actualmente,
            estamos trabajando en la implementación de gráficos que faciliten la interpretación de los datos.
        """)

# Sección de Implementación
if selected == 'Implementación':
    st.title(f"Conozca las Predicciones Realizadas")
    
    st.write("""
        En esta sección, puedes ver un resumen de las predicciones realizadas con el modelo de machine learning.
        Estos resultados se basan en los datos históricos y los patrones observados.
    """)
    
    if 'comparativo_2025_mg' in globals():
        st.subheader('Resumen de Predicciones')
        st.dataframe(comparativo_2025_mg.head())

# Sección de Realizar Predicción
if selected == 'Realizar Predicción':
    st.title(f"Obtenga la predicción de inscritos y matriculados - En construcción")
    st.write("""
        Esta sección permitirá realizar predicciones personalizadas de inscritos y matriculados con base en los datos
        históricos. Por ahora, estamos desarrollando esta funcionalidad.
    """)

# Sección de Generar Indicativo
if selected == 'Generar Indicativo':
    st.title(f"Obtenga el indicativo anual de grupos por centro de formación - En construcción")
    st.write("""
        En esta sección, podrás generar el indicativo anual de grupos por centro de formación. Actualmente, esta
        funcionalidad está en desarrollo y será implementada pronto.
    """)
