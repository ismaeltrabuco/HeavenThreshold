import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

st.title("🌌 Heaven Threshold — Modelo Agnóstico")

uploaded_file = st.file_uploader("Envie seu CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Dados carregados:", df.head())

    # Usuário escolhe qual é a coluna alvo
    target_col = st.selectbox("Escolha a variável alvo (y)", df.columns)

    # Features são todas as outras colunas numéricas
    features = [col for col in df.columns if col != target_col]

    if len(features) == 0:
        st.error("Não há features numéricas suficientes para treinar.")
    else:
        X = df[features]
        y = df[target_col]

        # Divide os dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Pipeline genérico: normalização + modelo
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200))
        ])

        # Treina
        pipe.fit(X_train, y_train)

        # Avalia
        score = pipe.score(X_test, y_test)
        st.write(f"✅ Acurácia no teste: {score:.2f}")

        # Coeficientes para legenda genérica
        if hasattr(pipe.named_steps["model"], "coef_"):
            coefs = pipe.named_steps["model"].coef_[0]
            legenda = {f"feat_{i}": f"{features[i]} (peso={coefs[i]:.2f})"
                       for i in range(len(features))}
            st.write("📌 Legenda genérica das features:")
            st.json(legenda)

        # Relatório
        y_pred = pipe.predict(X_test)
        st.text("📑 Relatório de classificação:")
        st.text(classification_report(y_test, y_pred))
