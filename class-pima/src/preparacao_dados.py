from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE #Para balancear os dados
from collections import Counter

def preparar_dados(dados_limpos):
    # separando X(atributos) e y (target)
    X = dados_limpos.drop(columns=['Outcome'])
    y = dados_limpos['Outcome']
    
    # split para o hold-out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # freq antes do SMOTE
    print(f'Frequência das classes ANTES do SMOTE: {Counter(y_train)}')
    
    # isntanciando o SMOTE com random_state e aplicando no treino
    balanceador = SMOTE(random_state=42)
    X_train, y_train = balanceador.fit_resample(X_train, y_train)
    
    # freq depois do SMOTE
    print(f'Frequência das classes DEPOIS do SMOTE: {Counter(y_train)}\n')
    
    return X_train, X_test, y_train, y_test