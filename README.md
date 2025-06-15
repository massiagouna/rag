# Analyse de documents

Ce projet propose une interface pour charger des documents pour constituer une base de connaissance qui pourra être questionnée avec un grand modèle de langage (_LLM_).
# RAG UI - Analyse de documents avec IA

Cette application permet d'analyser des documents PDF à l'aide d'une IA (Azure OpenAI) via Streamlit. Deux frameworks sont disponibles : **LangChain** et **LlamaIndex**.

## ⚙️ Étapes d'installation


0. **Assurez vous d'avoir une version récente de python (version 3.13 utilisée pour le projet)**

1. **Télécharger le dossier du projet**

   Clonez ou téléchargez ce dépôt sur votre machine.

2. **Ajouter un fichier de configuration**

   Créez un dossier `secrets/` à la racine du projet.  
   À l’intérieur, ajoutez un fichier `config.yaml` contenant vos clés Azure OpenAI :

   ```yaml
   chat:
     azure_endpoint: "https://..."
     azure_deployment: "..."
     azure_api_key: "..."
     azure_api_version: "..."

   embedding:
     azure_endpoint: "https://..."
     azure_deployment: "..."
     azure_api_key: "..."
     azure_api_version: "..."

3. **Installer les dépendances**
Dans un terminal, placez-vous à la racine du projet puis exécutez :

pip install -r requirements.txt


4. **Lancer l'application**

streamlit run app.py
