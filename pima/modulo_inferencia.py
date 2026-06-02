from pickle import load

def carregar_pkl(nome):
    return load(open(f'../models/{nome}.pkl', 'rb'))

def inferir_rf(dados_novos):
    modelo = carregar_pkl('classificador_rf')
    
    return modelo.predict(dados_novos)

def inferir_svm(dados_novos):
    modelo = carregar_pkl('classificador_svm')
    escalador = carregar_pkl('escalador_svm')
    dados_escalados = escalador.transform(dados_novos)
    
    return modelo.predict(dados_escalados)

def inferir_knn(dados_novos):
    modelo = carregar_pkl('classificador_knn')
    escalador = carregar_pkl('escalador_knn')
    dados_escalados = escalador.transform(dados_novos)
    
    return modelo.predict(dados_escalados)