from src.carregar_dados import carregar_dados
from src.pre_processamento import pre_processar
from src.preparacao_dados import preparar_dados
from src.avaliacao import avaliador
from src.treinamento_rf import treinar_rf
from src.treinamento_svm import treinar_svm
from src.treinamento_knn import treinar_knn
from src.salvar_modelo import salvar_modelo

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
    
    # treinando com o modelo rf
    print(f'\nTreinando com o Random Forest:')
    rf_treinado = treinar_rf(X_train, y_train)
    salvar_modelo(rf_treinado, 'classificador_rf')
    
    # treinando com o modelo svm - modelo e scaler salvos dentro da função
    print(f'\nTreinando com o Support Vector Machine:')
    svm_treinado, X_test_escalado_svm = treinar_svm(X_train, y_train, X_test)
    
    # treinando com o modelo knn - modelo e scaler salvos dentro da função
    print(f'\nTreinando com KNN - K-Nearest Neighbors:')
    knn_treinado, X_test_escalado_knn = treinar_knn(X_train, y_train, X_test)
    
    # avaliações - svm e knn usam o X_test escalado pelo respectivo scaler
    avaliador(rf_treinado, X_test, y_test, 'modelo Random Forest')
    avaliador(svm_treinado, X_test_escalado_svm, y_test, 'modelo Support Vector Machine')
    avaliador(knn_treinado, X_test_escalado_knn, y_test, 'modelo K-Nearest Neighbors')