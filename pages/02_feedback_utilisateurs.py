import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Feedback des utilisateurs",
    page_icon="📋"
)

st.title("📋 Feedback des utilisateurs")
st.subheader("Visualisez les retours laissés par les utilisateurs sur les réponses générées.")

DB_PATH = "feedback.db"

def get_all_feedbacks():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM feedbacks ORDER BY id DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture de la base de données : {e}")
        return pd.DataFrame()

def delete_all_feedbacks():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM feedbacks")
        conn.commit()
        conn.close()
        st.success("Tous les feedbacks ont été supprimés.")
    except Exception as e:
        st.error(f"Erreur lors de la suppression : {e}")

# Affichage
feedback_df = get_all_feedbacks()

if feedback_df.empty:
    st.info("Aucun feedback utilisateur enregistré pour le moment.")
else:
    st.dataframe(feedback_df, use_container_width=True)

    with st.expander("⚠️ Supprimer tous les feedbacks"):
        if st.button("Supprimer tous les feedbacks", type="primary"):
            delete_all_feedbacks()
