#from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_aws import BedrockEmbeddings# depreciated..using new


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
