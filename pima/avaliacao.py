import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def avaliador(modelo, X_test, y_test, nome_modelo='modelo'):
    y_pred = modelo.predict(X_test)
    
    # acurácia
    acuracia = accuracy_score(y_test, y_pred)
    
    # sensibilidade e especificidade via matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensibilidade = tp / (tp + fn)
    especificidade = tn / (tn + fp)
    
    print(f'\n--- Avaliação: {nome_modelo} ---')
    print(f'Acurácia: {acuracia:.4f}')
    print(f'Sensibilidade: {sensibilidade:.4f}')
    print(f'Especificidade: {especificidade:.4f}')
    
    # matriz de confusão visual
    ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test)
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.savefig(f'../plots/Pima_{nome_modelo}_confusion_matrix.png')
    plt.close()
    
    return{
        'acuracia': acuracia,
        'sensibilidade': sensibilidade,
        'especificidade': especificidade
    }