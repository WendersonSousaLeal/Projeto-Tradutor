"""
train_model.py
=====================================================================
Treina um RandomForestClassifier para reconhecer letras do alfabeto
manual de Libras a partir dos landmarks normalizados coletados por
collect_data.py.

Etapas:
    1. Carrega o CSV de dataset (pandas).
    2. Separa features (X) e rótulos (y), codifica as letras em inteiros.
    3. Divide em treino/teste de forma estratificada.
    4. Treina o RandomForestClassifier.
    5. Avalia com validação cruzada + relatório de classificação.
    6. Salva o modelo treinado + o encoder de rótulos com joblib.

Uso:
    python train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "landmarks_dataset.csv")
MODEL_OUT_PATH = os.path.join(BASE_DIR, "models", "libras_rf_model.pkl")

MIN_SAMPLES_PER_LETTER = 60  # aviso caso alguma letra tenha poucas amostras


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset não encontrado em '{path}'. Rode collect_data.py primeiro "
            "para gerar os dados de treinamento."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("O arquivo de dataset está vazio.")
    return df


def main():
    df = load_dataset(DATA_PATH)

    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1] - 1} features.")
    counts = df["label"].value_counts().sort_index()
    print("\nDistribuição de amostras por letra:")
    print(counts)

    letras_com_poucas_amostras = counts[counts < MIN_SAMPLES_PER_LETTER].index.tolist()
    if letras_com_poucas_amostras:
        print(
            f"\n[Aviso] As letras {letras_com_poucas_amostras} têm menos de "
            f"{MIN_SAMPLES_PER_LETTER} amostras. Considere coletar mais dados "
            "para elas antes de confiar no modelo em produção."
        )

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    # RandomForestClassifier trabalha bem com rótulos inteiros; o LabelEncoder
    # também nos dá um jeito fácil de voltar da predição (int) para a letra.
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Hiperparâmetros ---
    # n_estimators alto + min_samples_leaf pequeno ajudam o modelo a capturar
    # diferenças sutis entre sinais parecidos (ex.: M, N, S, T em Libras
    # diferem principalmente na posição do polegar). class_weight="balanced"
    # compensa letras com menos amostras coletadas do que outras.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\nAcurácia média (5-fold CV, conjunto de treino): "
          f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    y_pred = model.predict(X_test)
    print("\n=== Relatório de Classificação (conjunto de teste, dados nunca vistos) ===")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Quais landmarks/eixos mais pesam na decisão do modelo - útil para
    # diagnosticar se a normalização/coleta está fazendo sentido.
    feature_names = df.drop(columns=["label"]).columns
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features mais importantes:")
    for idx in top_idx:
        print(f"  {feature_names[idx]:>6s}: {importances[idx]:.4f}")

    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    joblib.dump({"model": model, "label_encoder": encoder}, MODEL_OUT_PATH)
    print(f"\nModelo salvo em: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
