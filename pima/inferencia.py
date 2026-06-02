import pandas as pd
from modulo_inferencia import inferir_rf, inferir_svm, inferir_knn

if __name__ == "__main__":
    # paciente novo fictício, com as 8 features na mesma ordem do treino
    paciente_novo = pd.DataFrame([{
        'Pregnancies': 2,
        'Glucose': 145,
        'BloodPressure': 70,
        'SkinThickness': 30,
        'Insulin': 125,
        'BMI': 33.5,
        'DiabetesPedigreeFunction': 0.45,
        'Age': 41
    }])

    # 0 = não diabético | 1 = diabético
    print('--- Inferência: paciente novo ---')
    print(f'RF  prevê:  {inferir_rf(paciente_novo)}')
    print(f'SVM prevê:  {inferir_svm(paciente_novo)}')
    print(f'KNN prevê:  {inferir_knn(paciente_novo)}')