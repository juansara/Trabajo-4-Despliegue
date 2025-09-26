import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight

from xgboost import XGBClassifier
from scipy.stats import randint, uniform

# 1) Cargar y limpiar
df = pd.read_csv("dataset.csv")

# columnas mínimas esperadas:
# artists, track_name, popularity, danceability, energy, loudness, speechiness,
# acousticness, instrumentalness, liveness, valence, tempo, duration_ms,
# time_signature, mode, explicit

# Limpieza básica
df = df.copy()
df = df[df["popularity"] > 0]                  # quitar ruido extremo
df["explicit"] = df["explicit"].astype(int)    # booleans -> int
df["artists"] = df["artists"].fillna("").astype(str)
df["track_name"] = df["track_name"].fillna("").astype(str)

# 2) Etiqueta (ajusta el umbral si quieres; >50 funciona bien)
df["target"] = (df["popularity"] > 50).astype(int)

# 3) Definir features
text_cols = ["artists", "track_name"]
num_cols = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
    "time_signature", "mode", "explicit"
]

X = df[text_cols + num_cols]
y = df["target"].values

# 4) Split estratificado (hold-out para medir de verdad)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5) Preprocesamiento: TF-IDF (unigramas y bigramas) + numéricas
preprocess = ColumnTransformer(
    transformers=[
        ("artists_tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=2), "artists"),
        ("track_tfidf",   TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=2), "track_name"),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
    n_jobs=-1
)

# 6) Desbalance: scale_pos_weight (relación entre clases en TRAIN)
pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

# 7) Modelo XGB (API sklearn para usar con Pipeline)
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=600,           # upper bound; early stopping implícito con regularización + LR
    tree_method="hist",         # rápido
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", xgb)
])

# 8) Búsqueda aleatoria de hiperparámetros (fuerte pero eficiente)
param_distributions = {
    "clf__learning_rate": uniform(0.01, 0.19),      # [0.01, 0.20]
    "clf__max_depth": randint(4, 11),               # [4..10]
    "clf__min_child_weight": randint(1, 7),         # [1..6]
    "clf__subsample": uniform(0.6, 0.4),            # [0.6..1.0]
    "clf__colsample_bytree": uniform(0.6, 0.4),     # [0.6..1.0]
    "clf__gamma": uniform(0.0, 5.0),                # [0..5]
    "clf__reg_lambda": uniform(0.0, 5.0),           # L2
    "clf__reg_alpha": uniform(0.0, 1.0),            # L1
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=35,                  # sube a 50 si quieres exprimir más
    scoring="accuracy",
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 9) Entrenar (el preprocessing está dentro del pipeline)
search.fit(X_train, y_train)

print("\nMejores hiperparámetros:")
print(search.best_params_)
print(f"CV best accuracy: {search.best_score_:.4f}")

# 10) Evaluar en TEST hold-out
best_model = search.best_estimator_
y_pred = (best_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy en TEST (hold-out): {acc:.4f}")
print("\nClassification report (TEST):")
print(classification_report(y_test, y_pred, digits=4))
print("\nMatriz de confusión (TEST):")
print(confusion_matrix(y_test, y_pred))

# 11) Guardar el pipeline completo (vectorizadores + modelo)
joblib.dump(best_model, "modelo_spotify_pipeline_xgb.pkl")
print("\n💾 Guardado: modelo_spotify_pipeline_xgb.pkl")
