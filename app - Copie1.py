import os
import tempfile
import streamlit as st
import pandas as pd

# Importer les deux frameworks
import rag.langchain as lc
import rag.llamaindex as li

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

# Initialiser les états
if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []
if 'last_response' not in st.session_state:
    st.session_state['last_response'] = ""
if 'last_question' not in st.session_state:
    st.session_state['last_question'] = ""
if 'last_feedback' not in st.session_state:
    st.session_state['last_feedback'] = ""
if 'feedback_choice' not in st.session_state:
    st.session_state['feedback_choice'] = ""

def main():
    st.title("Analyse de documents")
    st.subheader("Analysez vos documents avec une IA en les chargeant dans l'application. Puis posez toutes vos questions.")

    # Sélecteur de framework
    framework = st.radio("Choisissez le framework d'indexation :", ("langchain", "llamaindex"))
    module = lc if framework == "langchain" else li

    # Sélecteur du nombre de documents similaires
    top_k = st.slider("Nombre de documents similaires à récupérer :", min_value=1, max_value=10, value=5)

    # Rechargement des documents si changement de framework
    if 'last_framework' in st.session_state and st.session_state['last_framework'] != framework:
        st.warning("Framework changé : merci de recharger les documents.")
        st.session_state['stored_files'] = []
    st.session_state['last_framework'] = framework

    # Upload
    uploaded_files = st.file_uploader("Déposez vos fichiers ici ou chargez-les", accept_multiple_files=True)
    file_info = []
    if uploaded_files:
        for f in uploaded_files:
            size_in_kb = len(f.getvalue()) / 1024
            file_info.append({"Nom du fichier": f.name, "Taille (KB)": f"{size_in_kb:.2f}"})
            if f.name.endswith('.pdf') and f.name not in st.session_state['stored_files']:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, "temp.pdf")
                with open(path, "wb") as outfile:
                    outfile.write(f.read())
                module.store_pdf_file(path, f.name)
                st.session_state['stored_files'].append(f.name)
        st.table(pd.DataFrame(file_info))

    # Suppression fichiers
    files_to_be_deleted = set(st.session_state['stored_files']) - {f['Nom du fichier'] for f in file_info}
    for name in files_to_be_deleted:
        st.session_state['stored_files'].remove(name)
        try:
            module.delete_file_from_store(name)
        except NotImplementedError:
            st.error(f"Suppression non supportée pour {framework}")

    # Choix de langue
    langue = st.selectbox("Choisissez la langue de réponse :", ["Français", "Anglais", "Espagnol", "Allemand"])
    map_langue = {"Français": "French", "Anglais": "English", "Espagnol": "Spanish", "Allemand": "German"}

    # Question
    question = st.text_input("Votre question ici", value=st.session_state['last_question'])

    if st.button("Analyser"):
        try:
            response = module.answer_question(question, map_langue[langue], k=top_k)
        except TypeError:
            response = module.answer_question(question)

        st.session_state['last_question'] = question
        st.session_state['last_response'] = response
        st.session_state['last_feedback'] = ""
        st.session_state['feedback_choice'] = ""

    # Affichage réponse
    st.text_area("Zone de texte, réponse du modèle", value=st.session_state['last_response'], height=200)

    # Choix du feedback
    feedback_choice = st.selectbox(
        "Comment évaluez-vous cette réponse ?",
        ["", "👍 Bonne", "👎 Mauvaise", "❓ Hors sujet"],
        index=["", "👍 Bonne", "👎 Mauvaise", "❓ Hors sujet"].index(st.session_state['feedback_choice']),
        key="feedback_select"
    )

    # Bouton pour envoyer le feedback
    if st.button("Envoyer le feedback"):
        if feedback_choice:
            st.session_state['last_feedback'] = feedback_choice
            st.session_state['feedback_choice'] = feedback_choice
            st.success("Merci pour votre retour !")
            print(f"💬 Feedback utilisateur : {feedback_choice}")

if __name__ == "__main__":
    main()
