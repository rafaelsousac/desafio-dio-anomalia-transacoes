"""
Detecção de Fraude em Transações de Cartão de Crédito
-------------------------------------------------------
Pipeline completo: exploração, tratamento de desbalanceamento,
treino de modelos (Logistic Regression, Random Forest, XGBoost),
avaliação, ajuste de hiperparâmetros e explicabilidade (SHAP).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap


# =============================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(URL)

# Proporção de fraudes vs. transações normais (dataset é altamente desbalanceado)
print("Distribuição da variável alvo (Class):")
print(df["Class"].value_counts(normalize=True))


# =============================================================
# 2. FEATURE ENGINEERING
# =============================================================
# Log-transform em "Amount" para reduzir o efeito de outliers/assimetria
df["Amount_log"] = np.log1p(df["Amount"])

# Padronização de "Amount" (média 0, desvio padrão 1)
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])


X = df.drop(columns=["Class", "Amount"])
y = df["Class"]


# =============================================================
# 3. SEPARAÇÃO TREINO / TESTE
# =============================================================
# stratify=y garante que a proporção de fraudes seja mantida
# igual em treino e teste (essencial em dados desbalanceados).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)


# =============================================================
# 4. MODELO BASE: REGRESSÃO LOGÍSTICA
# =============================================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n[Regressão Logística] Classification Report:")
print(classification_report(y_test, y_pred))

# Probabilidades da classe positiva (fraude), usadas nas curvas ROC e PR
y_probs = model.predict_proba(X_test)[:, 1]


# --- Curva ROC ---
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("Curva ROC - Regressão Logística")
plt.xlabel("Falso Positivo (FPR)")
plt.ylabel("Verdadeiro Positivo (TPR)")
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))


# --- Curva Precision-Recall ---
# Mais informativa que a ROC em problemas com classes muito desbalanceadas,
# pois foca no desempenho sobre a classe minoritária (fraude).
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Curva Precision-Recall - Regressão Logística")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# =============================================================
# 5. BALANCEAMENTO DE DADOS
# =============================================================

# --- Undersampling (reduz a classe majoritária) ---
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(len(fraudes), random_state=42)
df_under = pd.concat([fraudes, normais])


# --- Oversampling com SMOTE (gera exemplos sintéticos da classe minoritária) ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nDistribuição da classe após SMOTE (apenas no treino):")
print(y_train_res.value_counts(normalize=True))


# =============================================================
# 6. RANDOM FOREST (com dados balanceados via SMOTE)
# =============================================================
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",  # reforça ainda mais o peso da classe minoritária
    n_jobs=-1,                # usa todos os núcleos disponíveis (mais rápido)
    random_state=42,
)
rf.fit(X_train_res, y_train_res)

y_pred_rf = rf.predict(X_test)
print("\n[Random Forest] Classification Report:")
print(classification_report(y_test, y_pred_rf))


# =============================================================
# 7. PIPELINE (Scaler + Regressão Logística)
# =============================================================
# Encapsular pré-processamento e modelo em um Pipeline evita erros
# de vazamento de dados (data leakage) e facilita reuso/deploy.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)

y_pred_pipeline = pipeline.predict(X_test)
print("\n[Pipeline: Scaler + Regressão Logística] Classification Report:")
print(classification_report(y_test, y_pred_pipeline))


# =============================================================
# 8. AJUSTE DE THRESHOLD (ponto de corte da probabilidade)
# =============================================================
# Por padrão, o modelo classifica como "fraude" (1) quando a
# probabilidade > 0.5. Em problemas de fraude, muitas vezes vale
# reduzir esse limiar para aumentar o recall (detectar mais fraudes),
# aceitando mais falsos positivos em troca.
threshold = 0.3
y_pred_custom = (y_probs > threshold).astype(int)

print(f"\n[Regressão Logística - Threshold customizado ({threshold})] Classification Report:")
print(classification_report(y_test, y_pred_custom))


# =============================================================
# 9. MODELO AVANÇADO: XGBOOST
# =============================================================
# scale_pos_weight ajuda a compensar o desbalanceamento sem precisar
# de oversampling — é uma alternativa ao SMOTE usada aqui para
# comparação de estratégias.
xgb = XGBClassifier(
    scale_pos_weight=10,
    eval_metric="logloss",
    random_state=42,
    # "use_label_encoder" foi removido: parâmetro depreciado em
    # versões recentes do XGBoost e não é mais necessário.
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
print("\n[XGBoost] Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# =============================================================
# 10. IMPORTÂNCIA DAS VARIÁVEIS (XGBoost)
# =============================================================
# Ajuda a entender quais variáveis mais influenciam a decisão do modelo.
importancias = xgb.feature_importances_

plt.figure(figsize=(10, 4))
plt.bar(X_train.columns, importancias)
plt.title("Importância das Variáveis - XGBoost")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# =============================================================
# 11. AJUSTE DE HIPERPARÂMETROS (GridSearchCV)
# =============================================================
# Testa combinações de hiperparâmetros para encontrar o modelo
# que maximiza o recall (métrica prioritária em detecção de fraude,
# já que deixar passar uma fraude custa mais caro que um falso alarme).

param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100],
}

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid,
    scoring="recall",
    cv=3,
)
grid.fit(X_train, y_train)

print("\nMelhores hiperparâmetros encontrados:", grid.best_params_)


# =============================================================
# 12. EXPLICABILIDADE (SHAP)
# =============================================================
# SHAP mostra, para cada variável, o quanto e em que direção ela
# empurra a previsão do modelo (em direção à fraude ou não-fraude).
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test[:100]) 

shap.plots.bar(shap_values)
