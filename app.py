import streamlit as st
import pandas as pd

st.title("✨ Heaven Threshold ✨")
st.write("""
A cosmic model of ascension based on the celestial formula.

We developed a formula that blends solidarity and human warmth to evaluate aspirants aiming to ascend to a higher plane.

Translated into the fields of business or personal growth, the model can assess the ability of a parameter, decision, or even a person to cross the threshold between two states:

**1 (ascension)** → success, passage, achievement.  
**-1 (non-passage)** → learning, reevaluation, a different path.

In our app, this process is represented as entry into a celestial event, but the same logic can be applied to various analytical contexts. 🚀
""")


uploaded_file = st.file_uploader("Envie seu dataset.csv", type="csv")

# Caso o usuário não envie nada, carrega o dataset padrão
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Nenhum arquivo enviado. Usando dataset cósmico padrão 🌌")
    df = pd.DataFrame({
        "aspirante": ["Terra_1","Terra_2","Sirius_1","Nibiru_1","Venus_1","Terra_3","Sirius_2","Nibiru_2","Venus_2","Terra_4"],
        "x": [0.3, -0.5, 0.8, -1.2, 1.5, -0.7, 0.9, -0.4, 1.1, 0.2],
        "y": [1 if v + 0.82 > 0 else -1 for v in [0.3, -0.5, 0.8, -1.2, 1.5, -0.7, 0.9, -0.4, 1.1, 0.2]]
    })

st.subheader("🔭 Dataset")
st.dataframe(df)

# Visualização básica
st.subheader("📊 Distribuição dos aspirantes")
st.bar_chart(df["x"])
