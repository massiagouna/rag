# Analyse de documents

Ce projet propose une interface pour charger des documents pour constituer une base de connaissance qui pourra être questionnée avec un grand modèle de langage (_LLM_).
# RAG UI - Analyse de documents avec IA

Cette application permet d'analyser des documents PDF à l'aide d'une IA (Azure OpenAI) via Streamlit. Deux frameworks sont disponibles : **LangChain** et **LlamaIndex**.

## ⚙️ Étapes d'installation


0. **Assurez vous d'avoir une version récente de python (version 3.13 utilisée pour le projet)**

1. **Télécharger le dossier du projet**

   Clonez ou téléchargez ce dépôt sur votre machine.

2. **Ajouter un fichier de configuration**

 Ajouter les clés d’API dans .streamlit/secrets.toml
crée un dossier .streamlit/ à la racine du projet, puis un fichier secrets.toml contenant tes clés Azure OpenAI :

[embedding]
azure_endpoint = "https://<TON_COMPTE>.openai.azure.com/"

azure_deployment = "nom-du-deployment-pour-les-embeddings"

azure_api_key = "clé-api-pour-les-embeddings"

azure_api_version = "2024-02-15-preview"

[chat]
azure_endpoint = "https://<TON_COMPTE>.openai.azure.com/"

azure_deployment = "nom-du-deployment-pour-le-chat"

azure_api_key = "clé-api-pour-le-chat"

azure_api_version = "2024-02-15-preview"


3. **Installer les dépendances**
Dans un terminal, placez-vous à la racine du projet puis exécutez :

pip install -r requirements.txt


4. **Lancer l'application**

streamlit run app.py
