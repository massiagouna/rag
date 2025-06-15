import pandas as pd
import streamlit as st

from rag.langchain import inspect_vector_store
from rag.langchain import get_vector_store_info 

st.set_page_config(
    page_title="Knowledge Base",
    page_icon="🧠",
)

st.title("Knowledge Base")
st.subheader("Visualiser les informations contenues dans la base de connaissances")

# Infos générales sur la base
infos = pd.DataFrame.from_dict(get_vector_store_info(), orient='index').transpose()
st.table(infos)

# Aperçu des documents (jusqu'à 100)
docs_df = inspect_vector_store(100)
st.subheader("Contenu indexé (jusqu’à 100 premiers chunks)")
st.dataframe(docs_df)
