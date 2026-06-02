from pickle import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from salvar_modelo import salvar_classificador

def treinar_knn(X_train, y_train, X_test):
    # definindo os hiperparâmetros
    n_neighbors = [3, 4, 5, 6, 7]
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan']
    random_state = 42
    n_iter = 10
    cv = 10
    verbose = 2
    n_jobs = -1
    
    # instanciando o scaler
    escalador = StandardScaler()
    
    dados_escalados = escalador.fit_transform(X_train)
    X_test_escalado = escalador.transform(X_test)
    
    # instanciando o estimador
    classificador_knn = KNeighborsClassifier()
    
    knn_grid = {
        'n_neighbors': n_neighbors,
        'weights': weights,
        'metric': metric,
    }
    
    hiperparametros_knn = RandomizedSearchCV(
        estimator=classificador_knn,
        param_distributions=knn_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state
    )
        
    hiperparametros_knn.fit(dados_escalados, y_train)
    
    # salvando o modelo e o scaler (o scaler é necessário para a inferência futura)
    salvar_classificador(hiperparametros_knn.best_estimator_, 'classificador_knn')
    salvar_classificador(escalador, 'escalador_knn')

    return hiperparametros_knn.best_estimator_, X_test_escalado