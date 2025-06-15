import yaml
from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

config = read_config("secrets/config.yaml")

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
    api_key=config["chat"]["azure_api_key"],
)

def get_meta_doc(extract: str) -> str:
    messages = [
        ("system", "You are a librarian extracting metadata from documents."),
        ("user", f"""Extract from the content the following metadata.
        Answer 'unknown' if you cannot find or generate the information.
        Metadata list:
        - title
        - author
        - source
        - type of content (e.g. scientific paper, litterature, news, etc.)
        - language
        - themes as a list of keywords

        <content>
        {extract}
        </content>
        """),
    ]
    response = llm.invoke(messages)
    return response.content

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool = True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(docs)

    for split in all_splits:
        split.metadata = {
            'document_name': doc_name,
            'insert_date': datetime.now()
        }

    if use_meta_doc and all_splits:
        extract = '\n\n'.join([split.page_content for split in all_splits[:min(10, len(all_splits))]])
        meta_doc = Document(
            page_content=get_meta_doc(extract),
            metadata={
                'document_name': doc_name,
                'insert_date': datetime.now()
            }
        )
        all_splits.append(meta_doc)

    vector_store.add_documents(documents=all_splits)

def delete_file_from_store(name: str) -> int:
    ids_to_remove = []
    for (id, doc) in vector_store.store.items():
        if name == doc['metadata']['document_name']:
            ids_to_remove.append(id)
    vector_store.delete(ids_to_remove)
    return len(ids_to_remove)

def inspect_vector_store(top_n: int = 10) -> list:
    docs = []
    for index, (id, doc) in enumerate(vector_store.store.items()):
        if index < top_n:
            docs.append({
                'id': id,
                'document_name': doc['metadata']['document_name'],
                'insert_date': doc['metadata']['insert_date'],
                'text': doc['text']
            })
        else:
            break
    return docs

def get_vector_store_info():
    nb_docs = 0
    max_date, min_date = None, None
    documents = set()
    for (id, doc) in vector_store.store.items():
        nb_docs += 1
        if max_date is None or max_date < doc['metadata']['insert_date']:
            max_date = doc['metadata']['insert_date']
        if min_date is None or min_date > doc['metadata']['insert_date']:
            min_date = doc['metadata']['insert_date']
        documents.add(doc['metadata']['document_name'])
    return {
        'nb_chunks': nb_docs,
        'min_insert_date': min_date,
        'max_insert_date': max_date,
        'nb_documents': len(documents)
    }

def retrieve(question: str, k: int = 5):
    print(f"[DEBUG] Récupération des documents avec k = {k}")
    return vector_store.similarity_search(question, k=k)

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

def answer_question(question: str, language: str = "French", k: int = 5) -> str:
    docs = retrieve(question, k=k)
    if not docs:
        return "⚠️ Aucun document pertinent n'a été trouvé dans la base de vecteurs."

    docs_content = "\n\n".join(doc.page_content for doc in docs)
    print("Question:", question)
    print("------")
    for doc in docs:
        print("Chunk:", doc.metadata.get("document_name", "unknown"))
        print(doc.page_content)
        print("------")

    messages = build_qa_messages(question, docs_content, language)
    response = llm.invoke(messages)
    return response.content
