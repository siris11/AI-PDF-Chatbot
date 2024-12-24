__import__('pysqlite3')
import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Function to query the database and generate a response
def query_rag(query_text: str):
    # Prepare the DB    
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB for relevant documents
    results = db.similarity_search_with_score(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.4:
        return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?", []
        
    # Extract the context from the search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    #the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)

    # Use the OllamaLLM model to get a response
    model = OllamaLLM(model="mistral", format="json")
    response = model.invoke(prompt)
    if not response or response == "{}":
        return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?", []

    # Get sources of the response (document IDs)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    
    if isinstance(response,dict) and "model" in response:
          return "Sorry, I couldn’t retrieve an appropriate response to your question."
    return response.strip() if isinstance(response,str) else str(response), sources


            # ---------------------------------------------------------------------------------------------- #    
# --- FOOTER ---

footer = """<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #000000;
    color: black;
    text-align: center;
	}
</style>
<div class="footer">
<font color = 'white'>Developed with ❤ by Siri</font>
</div>
"""
st.markdown(footer, unsafe_allow_html=True) 
             

if __name__ == "__main__":
    main()

