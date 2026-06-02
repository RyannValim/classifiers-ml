from pickle import dump

def salvar_classificador(modelo, nome_modelo):
    dump(modelo, open(f'../models/Pima_{nome_modelo}.pkl', 'wb'))