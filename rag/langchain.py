import yaml
from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# === Chargement de la configuration ===
def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Erreur de lecture du fichier YAML : {e}")
            return {}

config = {
    "embedding": {
        "azure_endpoint": st.secrets["embedding.azure_endpoint"],
        "azure_deployment": st.secrets["embedding.azure_deployment"],
        "azure_api_key": st.secrets["embedding.azure_api_key"],
        "azure_api_version": st.secrets["embedding.azure_api_version"],
    },
    "chat": {
        "azure_endpoint": st.secrets["chat.azure_endpoint"],
        "azure_deployment": st.secrets["chat.azure_deployment"],
        "azure_api_key": st.secrets["chat.azure_api_key"],
        "azure_api_version": st.secrets["chat.azure_api_version"],
    }
}

# === Initialisation embeddings et LLM ===
embedder = AzureOpenAIEmbeddings(
    azure_endpoint=config["embedding"]["azure_endpoint"],
    azure_deployment=config["embedding"]["azure_deployment"],
    openai_api_version=config["embedding"]["azure_api_version"],
    api_key=config["embedding"]["azure_api_key"]
)

vector_store = InMemoryVectorStore(embedder)

llm = AzureChatOpenAI(
    azure_endpoint=config["chat"]["azure_endpoint"],
    azure_deployment=config["chat"]["azure_deployment"],
    openai_api_version=config["chat"]["azure_api_version"],
    api_key=config["chat"]["azure_api_key"]
)

# === Extraction de métadonnées ===
def get_meta_doc(extract: str) -> str:
    messages = [
        SystemMessage(content="You are a librarian extracting metadata from documents."),
        HumanMessage(content=f"""Extract from the content the following metadata.
Answer 'unknown' if you cannot find or generate the information.
Metadata list:
- title
- author
- source
- type of content (e.g. scientific paper, literature, news, etc.)
- language
- themes as a list of keywords

<content>
{extract}
</content>""")
    ]
    response = llm.invoke(messages)
    return response.content

# === Enregistrement du document ===
def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool = True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)

    for split in splits:
        split.metadata = {
            "document_name": doc_name,
            "insert_date": datetime.now()
        }

    if use_meta_doc and splits:
        extract = "\n\n".join(split.page_content for split in splits[:10])
        meta_doc = Document(
            page_content=get_meta_doc(extract),
            metadata={
                "document_name": doc_name,
                "insert_date": datetime.now()
            }
        )
        splits.append(meta_doc)

    vector_store.add_documents(documents=splits)

# === Suppression de document ===
def delete_file_from_store(name: str) -> int:
    ids_to_remove = []
    for id, doc in vector_store.store.items():
        if doc["metadata"]["document_name"] == name:
            ids_to_remove.append(id)
    vector_store.delete(ids_to_remove)
    return len(ids_to_remove)

# === Exploration du contenu ===
def inspect_vector_store(top_n: int = 10) -> list:
    docs = []
    for i, (id, doc) in enumerate(vector_store.store.items()):
        if i < top_n:
            docs.append({
                "id": id,
                "document_name": doc["metadata"]["document_name"],
                "insert_date": doc["metadata"]["insert_date"],
                "text": doc["text"]
            })
    return docs

def get_vector_store_info():
    nb_docs = 0
    max_date, min_date = None, None
    documents = set()
    for _, doc in vector_store.store.items():
        nb_docs += 1
        date = doc["metadata"]["insert_date"]
        if max_date is None or date > max_date:
            max_date = date
        if min_date is None or date < min_date:
            min_date = date
        documents.add(doc["metadata"]["document_name"])
    return {
        "nb_chunks": nb_docs,
        "min_insert_date": min_date,
        "max_insert_date": max_date,
        "nb_documents": len(documents)
    }

# === Recherche vectorielle ===
def retrieve(question: str, k: int = 5):
    print(f"[DEBUG] Recherche de documents similaires (top {k})...")
    return vector_store.similarity_search(question, k=k)

# === Construction du prompt multi-langue ===
def build_qa_messages(question: str, context: str, language: str) -> list:
    return [
        SystemMessage(content=f"You are an assistant for question-answering tasks."),
        SystemMessage(content=f"Always answer ONLY in {language}, without translating into any other language."),
        SystemMessage(content=f"""Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
{context}"""),
        HumanMessage(content=question)
    ]

# === Réponse à la question ===
def answer_question(question: str, language: str = "French", k: int = 5) -> str:
    docs = retrieve(question, k=k)
    if not docs:
        return "⚠️ Aucun document pertinent n'a été trouvé dans la base de vecteurs."

    context = "\n\n".join(doc.page_content for doc in docs)
    print("📌 Contexte sélectionné pour la réponse :")
    for doc in docs:
        print("🔹", doc.page_content[:200].replace("\n", " "), "...")

    messages = build_qa_messages(question, context, language)
    response = llm.invoke(messages)
    return response.content
