import yaml
from datetime import datetime

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None


config = read_config("secrets/config.yaml")

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

# Pour garder les vecteurs en mémoire entre appels
if "llama_store" not in globals():
    llama_store = SimpleVectorStore()
vector_store = llama_store


def store_pdf_file(file_path: str, doc_name: str):
    print(f"📄 Lecture du fichier : {file_path}")
    loader = PyMuPDFReader()
    documents = loader.load(file_path)
    print(f"📃 {len(documents)} document(s) chargé(s)")

    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_nodes = []

    for doc_index, doc in enumerate(documents):
        print(f"➡️ Découpage du document {doc_index}")
        chunks = text_parser.split_text(doc.text)
        print(f"   🧩 {len(chunks)} chunk(s) trouvé(s)")
        for chunk_index, chunk in enumerate(chunks):
            node = TextNode(text=chunk)
            node.metadata = {
                "document_name": doc_name,
                "insert_date": datetime.now().isoformat()
            }
            try:
                embedding = embedder.get_text_embedding(chunk)
                node.embedding = embedding
                print(f"     ✅ Embedding généré pour le chunk {chunk_index}")
            except Exception as e:
                print(f"     ❌ Erreur d'embedding: {e}")
            all_nodes.append(node)

    print(f"🗃️ Insertion de {len(all_nodes)} node(s) dans le vecteur store")
    vector_store.add(all_nodes)


def delete_file_from_store(name: str) -> int:
    raise NotImplementedError('Suppression non supportée pour llamaindex.')


def retrieve(question: str):
    print(f"🔍 Question posée : {question}")
    print("ℹ️ Impossible de compter les vecteurs dans SimpleVectorStore directement (pas d'accès public).")
    query_embedding = embedder.get_query_embedding(question)
    print(f"📐 Embedding de la question généré")

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        mode="default"
    )
    results = vector_store.query(query)
    print(f"📥 {len(results.nodes)} document(s) retrouvé(s)")
    return results.nodes


def build_qa_messages(question: str, context: str, language: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"Respond strictly in {language}."),
        ("system", f"""Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        {context}"""),
        ("user", question)
    ]


def answer_question(question: str, language: str) -> str:
    print("⚙️ Appel à `answer_question()`")
    docs = retrieve(question)

    if not docs:
        print("⚠️ Aucun document trouvé dans les résultats")
        return "⚠️ Aucun document pertinent n'a été trouvé dans la base de vecteurs."

    docs_content = "\n\n".join(doc.get_content(metadata_mode="all") for doc in docs)
    print("📝 Contexte extrait :")
    for doc in docs:
        print("   📄", doc.get_content()[:200].replace("\n", " ") + "...")

    messages = build_qa_messages(question, docs_content, language)
    print("✉️ Messages envoyés au LLM")
    response = llm.invoke(messages)
    print("✅ Réponse reçue")
    return response.content
