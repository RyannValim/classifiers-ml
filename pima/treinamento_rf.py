import numpy as np
from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def treinar_rf(X_train, y_train):
    # definindo os hiperparâmetros
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
    max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
    criterion = ['gini', 'entropy']
    min_samples_split = [2, 4, 6, 8, 10]
    min_samples_leaf = [1, 2, 4]    
    max_features = ['sqrt', 'log2']
    n_jobs = -1
    random_state = 42
    verbose = 2
    cv = 10
    n_iter = 10
    
    # distribuição de parâmetros em grid
    rf_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'criterion': criterion,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
    }

    # instanciando o estimador
    classificador_rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    
    # treinando a base inteira com cross validation
    hiperparametros_rf = RandomizedSearchCV(
        estimator=classificador_rf,
        param_distributions=rf_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state
    )

    hiperparametros_rf.fit(X_train, y_train)
    
    # devolve a melhor combinação de valores
    return hiperparametros_rf.best_estimator_