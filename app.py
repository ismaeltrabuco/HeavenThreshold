import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

st.title("🌌 Heaven Threshold — Modelo Híbrido e Explicável")

uploaded_file = st.file_uploader("Envie seu CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dados carregados:")
    st.write(df.head())

    # Usuário escolhe variável alvo
    target_col = st.selectbox("Escolha a variável alvo (y)", df.columns)

    # Features = todas menos a target
    feature_cols = [col for col in df.columns if col != target_col]

    # Separa tipos de dados
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    st.write(f"🔢 Features numéricas detectadas: {numeric_features}")
    st.write(f"🔤 Features categóricas detectadas: {categorical_features}")

    X = df[feature_cols]
    y = df[target_col]

    # Pré-processamento: normaliza numéricas, codifica categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Pipeline final
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=200))
    ])

    # Divide os dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Treina
    pipe.fit(X_train, y_train)

    # Avalia
    score = pipe.score(X_test, y_test)
    st.success(f"✅ Acurácia no teste: {score:.2f}")

    # Importância das features explicada
    model = pipe.named_steps["model"]

    if hasattr(model, "coef_"):
        # Pega nomes das features após transformação
        feature_names_num = numeric_features
        feature_names_cat = pipe.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
        feature_names = list(feature_names_num) + list(feature_names_cat)

        coefs = model.coef_[0]

        # Cria legenda genérica
        legenda = {f"feat_{i}": f"{feature_names[i]} (peso={coefs[i]:.2f})"
                   for i in range(len(feature_names))}

        st.subheader("📌 Legenda genérica das features:")
        st.json(legenda)

    # Relatório
    y_pred = pipe.predict(X_test)
    st.subheader("📑 Relatório de classificação:")
    st.text(classification_report(y_test, y_pred))
