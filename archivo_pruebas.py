import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np
import xlsxwriter

# Función de ejemplo para calcular un nuevo DataFrame
def calcular_nuevo_dataframe(df):
    # Realizar algún cálculo en el DataFrame original y crear uno nuevo
    nuevo_df = df.copy()
    nuevo_df["Nueva_Columna"] = nuevo_df["Columna1"] * 2  # Ejemplo de operación
    return nuevo_df

# Cargar el archivo inicial
df_random = pd.DataFrame(
    np.random.rand(5, 3),  # 5 filas, 3 columnas con valores aleatorios entre 0 y 1
    columns=["Columna1", "Columna2", "Columna3"]  # Nombres de las columnas
)
def Descargar_dataframe():
    if True:
        st.write("Datos originales:")
        st.dataframe(df_random)

        # Botón para calcular el nuevo DataFrame
        if st.button("Calcular nuevo DataFrame"):
            nuevo_df = calcular_nuevo_dataframe(df_random)
            st.write("Nuevo DataFrame calculado:")
            st.dataframe(nuevo_df)

            # Botón para descargar el nuevo DataFrame como Excel
            def descargar_excel(dataframe):
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    dataframe.to_excel(writer, index=False, sheet_name="Hoja1")
                output.seek(0)
                return output

            excel_data = descargar_excel(nuevo_df)
            st.download_button(
                label="Descargar como Excel",
                data=excel_data,
                file_name="nuevo_dataframe.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            