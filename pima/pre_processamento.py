import numpy as np

def pre_processar(dados):
    # substituindo os zeros falsos por NaN para as colunas problemáticas
    dados_limpos = dados.copy()
    colunas_problema = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    dados_limpos[colunas_problema] = dados_limpos[colunas_problema].replace(0, np.nan)

    # apontando quem são as colunas que serão preenchidas pela média e pela mediana
    colunas_media = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
    colunas_mediana = ['Insulin']
    
    # preenchendo os valores ausentes com a média e a mediana
    dados_limpos[colunas_media] = dados_limpos[colunas_media].fillna(dados_limpos[colunas_media].mean())
    dados_limpos[colunas_mediana] = dados_limpos[colunas_mediana].fillna(dados_limpos[colunas_mediana].median())

    return dados_limpos