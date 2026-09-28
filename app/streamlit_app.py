"""Interface Streamlit : upload d'une vidéo -> timeline des phases + historique."""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Assistant IA – analyse embryonnaire", layout="wide")
st.title("Assistant IA – analyse du développement embryonnaire")
st.caption("Prototype de démonstration. Outil d'aide : la validation par l'embryologiste reste obligatoire.")

tab1, tab2 = st.tabs(["Nouvelle analyse", "Historique"])

with tab1:
    video = st.file_uploader("Vidéo time-lapse", type=["avi", "mp4", "mov"])
    if video and st.button("Analyser"):
        with st.spinner("Analyse en cours..."):
            try:
                r = requests.post(f"{API_URL}/predict", files={"file": (video.name, video.getvalue())}, timeout=600)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                st.error(f"Erreur : {e}")
                data = None
        if data:
            df = pd.DataFrame(data["predictions"])
            df["frame_fin"] = df["frame"].shift(-1).fillna(df["frame"].max() + 1)
            st.metric("Confiance moyenne", f"{data['confiance_moyenne']:.1%}")
            chart = alt.Chart(df).mark_rect().encode(
                x=alt.X("frame:Q", title="Image"),
                x2=alt.X2("frame_fin:Q"),
                color=alt.Color("phase:N", title="Phase"),
                tooltip=["frame", "phase", alt.Tooltip("confiance:Q", format=".2f")],
            ).properties(height=90)
            st.altair_chart(chart.properties(width="container"), use_container_width=True)
            st.dataframe(df.drop(columns=["frame_fin"]), use_container_width=True)

with tab2:
    if st.button("Rafraîchir"):
        pass
    try:
        h = requests.get(f"{API_URL}/history", timeout=10).json()
        st.dataframe(pd.DataFrame(h), use_container_width=True)
    except Exception as e:
        st.warning(f"Historique indisponible : {e}")
