# 💳 Detecção de Fraude em Transações de Cartão de Crédito

Pipeline completo de Machine Learning para identificar transações fraudulentas em um dataset real e altamente desbalanceado, cobrindo desde a exploração dos dados até a explicabilidade do modelo final.

---

## 🎯 Sobre o projeto

Transações fraudulentas representam uma fração muito pequena do total (menos de 1% no dataset usado), o que torna esse um problema clássico de **classificação com classes desbalanceadas**. O projeto explora diferentes estratégias para lidar com esse desafio — desde ajuste de threshold até balanceamento de classes e modelos mais robustos como XGBoost — comparando o desempenho de cada abordagem.

### Objetivos
- Treinar e comparar modelos de classificação (Regressão Logística, Random Forest e XGBoost) para detecção de fraude.
- Lidar corretamente com o desbalanceamento de classes, evitando vazamento de dados (*data leakage*).
- Avaliar os modelos com métricas adequadas ao problema (Precision, Recall, AUC-ROC, Precision-Recall).
- Interpretar as decisões do modelo com **SHAP**, entendendo quais variáveis mais influenciam a previsão de fraude.

---

## 📊 Dataset

- **Fonte:** [Credit Card Fraud Dataset (TensorFlow Storage)](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)
- **Registros:** transações de cartão de crédito, cada uma marcada como fraude (`1`) ou normal (`0`) na coluna `Class`.
- **Variáveis:** a maioria das colunas (`V1` a `V28`) já vem anonimizada via PCA (por questões de privacidade), além de `Time` e `Amount` (valor da transação).
- **Desbalanceamento:** fraudes representam uma parcela mínima dos dados, o que exige cuidado especial no treino e na avaliação dos modelos.

---

## 🧠 Etapas do pipeline

O script (`deteccao_fraude.py`) está organizado em 12 seções sequenciais:

| # | Etapa | Descrição |
|---|-------|-----------|
| 1 | Carregamento dos dados | Leitura do CSV e análise da proporção de fraudes |
| 2 | Feature Engineering | Transformação log e padronização da variável `Amount` |
| 3 | Split treino/teste | Separação estratificada (mantém a proporção de fraudes) |
| 4 | Modelo base | Regressão Logística + curvas ROC e Precision-Recall |
| 5 | Balanceamento de dados | Undersampling e SMOTE (aplicado apenas no treino) |
| 6 | Random Forest | Treinado com dados balanceados via SMOTE |
| 7 | Pipeline | Scaler + Regressão Logística encapsulados |
| 8 | Ajuste de threshold | Teste de um limiar de decisão customizado (0.3) |
| 9 | XGBoost | Modelo avançado com `scale_pos_weight` para lidar com o desbalanceamento |
| 10 | Importância das variáveis | Quais features mais pesam na decisão do XGBoost |
| 11 | GridSearchCV | Busca pelos melhores hiperparâmetros, otimizando o Recall |
| 12 | SHAP | Explicabilidade: como cada variável influencia cada previsão |

---

## 🛠️ Tecnologias e bibliotecas

- **pandas** / **numpy** — manipulação e transformação de dados
- **scikit-learn** — modelos, métricas, pipeline e busca de hiperparâmetros
- **imbalanced-learn (SMOTE)** — balanceamento de classes
- **XGBoost** — modelo de gradient boosting
- **SHAP** — explicabilidade do modelo
- **matplotlib** — visualização de gráficos e curvas

---

## ▶️ Como executar

### 1. Instale as dependências
```bash
pip install -r requirements.txt
```

### 2. Execute o script
```bash
python deteccao_fraude.py
```

> ⚠️ O download do dataset é feito automaticamente via URL no início do script — é necessário estar conectado à internet na primeira execução.

---

## 📈 Métricas utilizadas

Em problemas de detecção de fraude, a métrica **acurácia** é enganosa (um modelo que sempre prevê "não é fraude" já acerta mais de 99%). Por isso, o projeto prioriza:

- **Recall (Sensibilidade):** de todas as fraudes reais, quantas o modelo conseguiu identificar. É a métrica mais crítica aqui — deixar passar uma fraude tem custo alto.
- **Precision:** de todas as transações marcadas como fraude, quantas realmente eram fraude (evita excesso de falsos alarmes).
- **AUC-ROC:** capacidade geral do modelo de separar as duas classes.
- **Curva Precision-Recall:** mais informativa que a ROC em datasets desbalanceados, pois foca no desempenho sobre a classe minoritária.

---

## ⚖️ Estratégias de balanceamento comparadas

| Estratégia | Como funciona | Onde é usada no script |
|---|---|---|
| **Undersampling** | Reduz a classe majoritária ao tamanho da minoritária | Gerado (`df_under`) para referência/exploração |
| **SMOTE (Oversampling)** | Cria exemplos sintéticos da classe minoritária | Usado no treino do Random Forest |
| **class_weight="balanced"** | Penaliza mais os erros na classe minoritária | Random Forest |
| **scale_pos_weight** | Ajusta o peso da classe positiva sem gerar dados sintéticos | XGBoost |
| **Threshold customizado** | Reduz o limiar de decisão (padrão 0.5 → 0.3) para aumentar o recall | Regressão Logística |

> ⚠️ **Cuidado importante aplicado no código:** todo balanceamento (SMOTE, undersampling) é feito **somente no conjunto de treino**. Aplicar essas técnicas antes do split ou no conjunto de teste distorce a avaliação do modelo (data leakage).

---

## 🔍 Explicabilidade (SHAP)

Após o treino do XGBoost, o script gera um gráfico SHAP que mostra **quais variáveis mais influenciam a decisão do modelo** e em qual direção (aumentando ou reduzindo a probabilidade de fraude). Isso é essencial em contextos reais de fraude, onde é preciso justificar por que uma transação foi bloqueada.

---

## 📁 Estrutura do projeto
```
├── deteccao_fraude.py   # Script principal com todo o pipeline
├── requirements.txt     # Dependências do projeto
└── README.md            # Este arquivo
```

---

## 🚀 Possíveis melhorias futuras
- Comparar métricas dos 4 modelos lado a lado em uma tabela consolidada.
- Testar outras técnicas de balanceamento (ADASYN, undersampling + oversampling combinados).
- Validação cruzada estratificada para todos os modelos, não só no GridSearch.
- Deploy do modelo final via API (ex: FastAPI) para uso em produção.
