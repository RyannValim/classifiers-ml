from carregamento_dados import carregar_dados
from pre_processamento import pre_processar
from preparacao_dados import preparar_dados

if __name__ == "__main__":
    # carrega os dados para a main
    dados = carregar_dados()
    
    # pre-processamento dos dados que apresentam colunas com valores ausentes
    dados_limpos = pre_processar(dados)
    
    # preparando os dados de treino/teste para o modelo
    X_train, X_test, y_train, y_test = preparar_dados(dados_limpos)
    
    print('Dados preparados com SMOTE:')
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")