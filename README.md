# classifiers-ml

Projeto de estudo de classificadores de machine learning usando scikit-learn. Cada base de dados é classificada por um ou mais modelos, com tratamento de dados ausentes, balanceamento de classes e busca de hiperparâmetros.

> Repositório de estudo, licenciado sob MIT.

## Estrutura do projeto

```
classifiers-ml/
├── bank/                  # classificador da base Bank Marketing (em construção)
├── datasets/              # bases de dados (.csv, .txt)
├── estudos/               # material de aula e enunciados
├── models/                # modelos treinados (.pkl)
├── pima/                  # classificador da base Pima Diabetes
├── plots/                 # matrizes de confusão e gráficos gerados
├── LICENSE
├── README.md
└── requirements.txt
```

## Configuração do ambiente

O projeto usa Python 3.12. Recomenda-se trabalhar dentro de um ambiente virtual.

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd classifiers-ml
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
# .venv\Scripts\activate      # Windows
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

As dependências principais são: scikit-learn, pandas, matplotlib e imbalanced-learn (imblearn).

## Classificador Pima Diabetes

Classifica a base Pima Indians Diabetes (768 instâncias, 8 atributos + alvo `Outcome`) usando três modelos: Random Forest, Support Vector Machine e K-Nearest Neighbors.

### Pipeline

O fluxo executado é:

1. **Carregamento** — leitura do CSV (`carregamento_dados.py`).
2. **Pré-processamento** — zeros inválidos viram NaN e são imputados por média (Glucose, BloodPressure, SkinThickness, BMI) ou mediana (Insulin) conforme a assimetria de cada distribuição (`pre_processamento.py`).
3. **Preparação** — separação de atributos/alvo, split treino/teste (hold-out, `test_size=0.3`) e balanceamento com SMOTE aplicado **somente no conjunto de treino**, depois do split, para evitar vazamento de dados (`preparacao_dados.py`).
4. **Treinamento** — cada modelo é treinado com `RandomizedSearchCV` para busca de hiperparâmetros. SVM e KNN normalizam os dados com `StandardScaler`, pois são sensíveis à escala das features (`treinamento_rf.py`, `treinamento_svm.py`, `treinamento_knn.py`).
5. **Avaliação** — cálculo de acurácia, sensibilidade e especificidade, além da matriz de confusão salva em `plots/` (`avaliacao.py`).

### Como rodar

A partir da pasta `pima/`:

```bash
cd pima
python main.py
```

A execução imprime no terminal as métricas de cada modelo, salva os modelos treinados em `models/` e as matrizes de confusão em `plots/`.

## Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
