import yaml
from datetime import datetime
import streamlit as st

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader

# === Paramètres de découpage ===
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# === Lecture de la configuration YAML ===
def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"❌ Erreur lecture YAML: {e}")
        return {}

config = read_config("secrets/config.yaml")

# === Initialisation LLM et Embedding ===
llm = AzureOpenAI(
    model=config["chat"]["azure_deployment"],
    deployment_name=config["chat"]["azure_deployment"],
    api_key=config["chat"]["azure_api_key"],
    azure_endpoint=config["chat"]["azure_endpoint"],
    api_version=config["chat"]["azure_api_version"],
)

embedder = AzureOpenAIEmbedding(
    model=config["embedding"]["azure_deployment"],
    deployment_name=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    azure_endpoint=config["embedding"]["azure_endpoint"],
    api_version=config["embedding"]["azure_api_version"],
)

Settings.llm = llm
Settings.embed_model = embedder

# === Initialisation du vecteur store partagé (mémorisé dans la session) ===
if "llama_store" not in st.session_state:
    print("🔄 Initialisation d'un nouveau vector store LlamaIndex")
    st.session_state["llama_store"] = SimpleVectorStore()
vector_store = st.session_state["llama_store"]

# === Indexation PDF ===
def store_pdf_file(file_path: str, doc_name: str):
    print(f"📄 Lecture du fichier : {file_path}")
    loader = PyMuPDFReader()
    documents = loader.load(file_path)
    print(f"📃 {len(documents)} document(s) chargé(s)")

    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = []

    for i, doc in enumerate(documents):
        print(f"➡️ Découpage du document {i}")
        chunks = parser.split_text(doc.text)
        print(f"   🧩 {len(chunks)} chunk(s) trouvé(s)")

        for j, chunk in enumerate(chunks):
            try:
                embedding = embedder.get_query_embedding(chunk)  
                node = TextNode(text=chunk)
                node.embedding = embedding
                node.metadata = {
                    "document_name": doc_name,
                    "insert_date": datetime.now().isoformat()
                }
                nodes.append(node)
                print(f"     ✅ Embedding généré pour le chunk {j}")
            except Exception as e:
                print(f"     ❌ Erreur embedding (chunk {j}): {e}")

    print(f"🗃️ Insertion de {len(nodes)} node(s) dans le vecteur store")
    vector_store.add(nodes)

# === Suppression (non supportée) ===
def delete_file_from_store(name: str):
    raise NotImplementedError("❌ Suppression non supportée pour LlamaIndex.")

# === Recherche vectorielle ===
def retrieve(question: str, k: int = 5):
    print(f"🔍 Question posée : {question}")
    query_embedding = embedder.get_query_embedding(question)
    print("📐 Embedding généré pour la question")

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=k,
        mode="default"
    )

    results = vector_store.query(query)

    if results is None or not results.nodes:
        print("⚠️ Aucun résultat trouvé dans le vector store.")
        return []

    print(f"📥 {len(results.nodes)} document(s) retrouvé(s)")
    return results.nodes

# === Construction du prompt ===
def build_qa_messages(question: str, context: str, language: str):
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"Respond strictly in {language}."),
        ("system", f"""Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
{context}"""),
        ("user", question)
    ]

# === Génération de réponse ===
def answer_question(question: str, language: str, k: int = 5) -> str:
    print("⚙️ Appel à `answer_question()`")
    docs = retrieve(question, k=k)

    if not docs:
        return "⚠️ Aucun document pertinent n'a été trouvé dans la base de vecteurs."

    context = "\n\n".join(
        doc.text if hasattr(doc, "text") else doc.get_content(metadata_mode="all")
        for doc in docs
    )

    print("📝 Contexte extrait :")
    for i, doc in enumerate(docs):
        preview = doc.text[:200].replace("\n", " ") if hasattr(doc, "text") else doc.get_content()[:200]
        print(f"   📄 {i}: {preview}...")

    messages = build_qa_messages(question, context, language)
    print("✉️ Envoi au LLM...")
    response = llm.invoke(messages)
    print("✅ Réponse reçue")
    return response.content
