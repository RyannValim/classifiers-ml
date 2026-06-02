from pickle import dump
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from salvar_modelo import salvar_classificador

def treinar_svm(X_train, y_train, X_test):
    # definindo os hiperparâmetros
    C = [0.1, 1, 10, 100, 1000]
    kernel = ['rbf', 'linear', 'poly']
    gamma = ['scale', 'auto', 0.001, 0.01, 0.1]
    random_state = 42
    n_iter = 10
    cv = 10
    verbose = 2
    n_jobs = -1
    
    # instancia o scaler
    escalador = StandardScaler()
    
    dados_escalados = escalador.fit_transform(X_train)
    X_test_escalado = escalador.transform(X_test)
    
    # instanciando o estimador
    classificador_svm = SVC(random_state=random_state)
    
    svm_grid = {
        'C': C,                 # parâmetro de regularização, quanto mais alto, mais rigoroso
        'kernel': kernel,       # define como o SVM mede a distância entre os pontos, linear: traça uma reta separando as classes | rbf: consegue traçar fronteiras curvas, mais complexas | poly: usa curvas polinomiais
        'gamma': gamma          # controla o raio de influência de cada ponto de treino, quanto mais algo, influencia somente os vizinhos mais próximos. Só afeta rbf e poly, no linear é ignorado
    }
    
    hiperparametros_svm = RandomizedSearchCV(
        estimator=classificador_svm,
        param_distributions=svm_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    hiperparametros_svm.fit(dados_escalados, y_train)
    
    # salvando o modelo e o scaler (o scaler é necessário para a inferência futura)
    salvar_classificador(hiperparametros_svm.best_estimator_, 'classificador_svm')
    salvar_classificador(escalador, 'escalador_svm')
    
    # retorna o modelo e o X_test_escalado
    return hiperparametros_svm.best_estimator_, X_test_escalado