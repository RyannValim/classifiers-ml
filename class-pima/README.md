# class-pima

Classificador de diabetes com a base Pima Indians Diabetes usando três modelos de machine learning: Random Forest, Support Vector Machine e K-Nearest Neighbors. Inclui pré-processamento, balanceamento de classes com SMOTE, busca de hiperparâmetros e módulo de inferência para novos pacientes.

> Repositório de estudo, licenciado sob MIT.

## Estrutura do projeto

```
class-pima/
├── data/
│   └── raw/
│       └── pima_diabetes.csv            # base de dados original
│   └── processed/
│       └── pima_diabetes-processed.csv  # base de dados processada
├── models/                     	 # modelos e escaladores salvos (.pkl)
├── plots/                      	 # matrizes de confusão geradas (.png)
├── src/
│   ├── carregar_dados.py      		 # leitura do CSV
│   ├── pre_processamento.py    	 # tratamento de valores ausentes
│   ├── preparacao_dados.py     	 # split treino/teste e balanceamento SMOTE
│   ├── treinamento_rf.py       	 # treinamento Random Forest
│   ├── treinamento_svm.py      	 # treinamento SVM + escalador
│   ├── treinamento_knn.py     	 	 # treinamento KNN + escalador
│   ├── avaliacao.py           	 	 # métricas e matriz de confusão
│   ├── salvar_modelo.py        	 # persistência em pickle
│   └── modulo_inferencia.py    	 # carrega modelos salvos para inferência
├── inferencia.py               	 # entrada para predição de novos pacientes
├── main.py                     	 # entrada para treinamento completo
├── requirements.txt
└── LICENSE
```

## Configuração do ambiente

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd class-pima
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

Dependências: `scikit-learn`, `pandas`, `matplotlib`, `imbalanced-learn`.

## Pipeline de treinamento

A base Pima Indians Diabetes tem 768 instâncias, 8 atributos numéricos e o alvo binário `Outcome` (0 = não diabético, 1 = diabético).

| Etapa              | Módulo                       | O que faz                                                                                                                                                       |
| ------------------ | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Carregamento       | `carregar_dados.py`         | Lê o CSV em DataFrame                                                                                                                                          |
| Pré-processamento | `pre_processamento.py`      | Substitui zeros inválidos por NaN e imputa: média para Glucose, BloodPressure, SkinThickness e BMI; mediana para Insulin                                      |
| Preparação       | `preparacao_dados.py`       | Split treino/teste (70/30,`random_state=42`), depois aplica SMOTE **somente no treino** para evitar data leakage                                        |
| Treinamento        | `treinamento_rf/svm/knn.py` | Cada modelo usa `RandomizedSearchCV` (`cv=10`, `n_iter=10`). SVM e KNN aplicam `StandardScaler` antes do treino e salvam o escalador junto com o modelo |
| Avaliação        | `avaliacao.py`              | Calcula acurácia, sensibilidade e especificidade; salva a matriz de confusão em `plots/`                                                                    |

## Como rodar

### Treinamento

A partir da raiz do projeto:

```bash
python main.py
```

Imprime as métricas de cada modelo no terminal, salva os modelos em `models/` e as matrizes de confusão em `plots/`.

Arquivos gerados em `models/`:

```
classificador_rf.pkl
classificador_svm.pkl   escalador_svm.pkl
classificador_knn.pkl   escalador_knn.pkl
```

### Inferência

Edite os atributos do paciente em `inferencia.py` e execute:

```bash
python inferencia.py
```

Saída esperada (valores variam conforme os modelos treinados):

```
--- Inferência: paciente novo ---
RF  prevê:  [0]
SVM prevê:  [0]
KNN prevê:  [0]
```

> Os modelos devem estar treinados antes de rodar a inferência.

## Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
