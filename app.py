import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="✨ Heaven Threshold ✨", layout="wide")

st.title("✨ Heaven Threshold ✨")
st.write("""
Um **modelo cósmico de ascensão** inspirado na fórmula celestial.

O modelo avalia aspirantes com base em **traços humanos e sociais** — amor, perdão, apoio familiar, vínculos afetivos e escolhas de vida.  
O objetivo é calcular a **probabilidade de ascensão (y=1)** e sugerir **intervenções prioritárias**. 🚀
""")

# Upload
uploaded_file = st.file_uploader("📂 Envie seu dataset.csv", type="csv")

# Dataset padrão com novas features
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Nenhum arquivo enviado. Usando dataset cósmico padrão 🌌")
    df = pd.DataFrame({
        "aspirante": ["Terra_1","Terra_2","Sirius_1","Nibiru_1","Venus_1",
                      "Terra_3","Sirius_2","Nibiru_2","Venus_2","Terra_4"],
        "x": [0.3, -0.5, 0.8, -1.2, 1.5, -0.7, 0.9, -0.4, 1.1, 0.2],
        "ama": [1,0,1,0,1,0,1,0,1,1],
        "perdoou": [0,1,1,0,1,0,1,1,1,0],
        "homicidio": [0,0,0,1,0,0,0,0,0,0],
        "family_support": [1,0,1,0,1,0,1,0,1,1],
        "has_partner": [1,0,1,0,1,0,0,1,1,0],
        "has_children": [0,0,1,0,1,0,1,0,0,1],
    })
    df["y"] = [1 if v + 0.82 > 0 else -1 for v in df["x"]]

# Mostra dataset
st.subheader("🔭 Dataset")
st.dataframe(df)

# ===============================
# Modelo logístico simples
# ===============================
features = ["x","ama","perdoou","homicidio","family_support","has_partner","has_children"]
X = df[features]
y = (df["y"] == 1).astype(int)

model = LogisticRegression()
model.fit(X, y)

# Probabilidades preditas
df["prob_ascensao"] = model.predict_proba(X)[:,1]

# ===============================
# Gráfico de importância das features
# ===============================
st.subheader("📊 Importância das features")
coef_df = pd.DataFrame({
    "feature": features,
    "peso": model.coef_[0]
}).sort_values(by="peso", ascending=False)

fig, ax = plt.subplots(figsize=(6,4))
ax.barh(coef_df["feature"], coef_df["peso"], color="skyblue")
ax.set_xlabel("Peso (coeficiente logístico)")
ax.set_title("Importância relativa das features")
st.pyplot(fig)

# ===============================
# Sugestões de intervenção
# ===============================
st.subheader("💡 Sugestões de Intervenção")

def sugerir(row):
    if row["homicidio"] == 1:
        return "⚠️ Prevenção de violência urgente"
    elif row["family_support"] == 0:
        return "👨‍👩‍👧 Fortalecer apoio familiar"
    elif row["perdoou"] == 0:
        return "💙 Trabalhar perdão"
    elif row["ama"] == 0:
        return "💞 Incentivar amar"
    else:
        return "✨ Manter equilíbrio cósmico"

df["intervencao"] = df.apply(sugerir, axis=1)

st.dataframe(df[["aspirante","prob_ascensao","intervencao"]])

# ===============================
# Visualização final
# ===============================
st.subheader("🌌 Distribuição de probabilidades de ascensão")
st.bar_chart(df.set_index("aspirante")["prob_ascensao"])
