from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import yaml

config = yaml.safe_load(open("secrets/config.yaml"))

llm = AzureChatOpenAI(
    azure_endpoint=config["chat"]["azure_endpoint"],
    azure_deployment=config["chat"]["azure_deployment"],
    openai_api_version=config["chat"]["azure_api_version"],
    api_key=config["chat"]["azure_api_key"],
)

messages = [
    SystemMessage(content="Always answer ONLY in Spanish, without translating into any other language."),
    HumanMessage(content="What is artificial intelligence?")
]

response = llm.invoke(messages)
print("🟡 Réponse du modèle :", response.content)
