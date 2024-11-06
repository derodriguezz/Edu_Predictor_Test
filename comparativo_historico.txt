import pandas as pd

# Leer el archivo comparativo_2025_mg.csv
comparativo_2025_mg = pd.read_csv('./comparativo_2025_mg.csv', encoding='utf-8')

# Leer el archivo comparativo_historico_mg.csv
comparativo_historico_mg = pd.read_csv('./comparativo_historico_mg.csv', encoding='utf-8')

#comparativo_historico_mg.rename(columns={'AÑO_FICHA': 'ANO_FICHA'}, inplace=True)

# Filtrar comparativo_historico_mg por AÑO_FICHA igual a 2024
#comparativo_2024 = comparativo_historico_mg[comparativo_historico_mg['AÑO_FICHA'] == 2024]

#-----------------------------------------------------------------------------------------------
