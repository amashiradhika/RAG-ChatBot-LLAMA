#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import numpy as np
import faiss
import pandas as pd
from langchain.chains import LLMChain
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document

# Initialize your LLM model
chat_model = ChatTogether(
    together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
    model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
)

# Prompt engineering: Make the context more specific and improve the instructions
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert financial advisor. Use the context to answer questions accurately and concisely.\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer (be specific and avoid hallucinations):"
    )
)

# Load the embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Example dummy text (can be replaced with actual data)
dummy_text = ["""
Bajaj customers: Interest rates range from 7.16% to 8.40% per annum (p.a.)
- Senior citizens: Interest rates range from 7.39% to 8.65% p.a., with an additional rate benefit of up to 0.40% p.a.
- Non-Resident Indians (NRIs): Interest rates of up to 8.35% p.a.
- Systematic Deposit Plan (SDP): A regular monthly deposit scheme
Bajaj Finance FDs are considered to be safe and secure, with an AAA(Stable) rating from ICRA and AAA/STABLE from CRISIL.
The minimum investment amount is Rs. 15,000, and the tenure can range from 12 to 60 months.
Before booking an FD, it's important to check the eligibility criteria and document requirements on the lender's website.
"""]

# Create embeddings
embeddings = embedding_model.embed_documents(dummy_text)
embeddings = np.array(embeddings)
dimension = embeddings.shape[1]

# Initialize FAISS index and add embeddings
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Setup the docstore and mapping between index and documents
docstore = InMemoryDocstore({0: Document(page_content=dummy_text[0])})
index_to_docstore_id = {0: 0}

# Initialize the FAISS vector store
vector_store = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embedding_model.embed_query  # Use the embedding function
)

# Define the QA chain for answering questions
qa_chain = LLMChain(
    llm=chat_model,
    prompt=prompt_template
)

# Example usage
def get_answer(question):
    # Create an embedding for the query
    query_embedding = embedding_model.embed_query(question)
    
    # Search the vector store for the most similar documents
    D, I = index.search(np.array([query_embedding]), k=1)
    
    # Retrieve the most relevant document
    doc_id = I[0][0]
    document = docstore.search(doc_id)
    context = document.page_content
    
    # Use the qa_chain to get an answer
    answer = qa_chain.run(context=context, question=question, clean_up_tokenization_spaces=False)
    
    return answer

if __name__ == "__main__":
    # Define a test question
    question = "What is the interest rate for senior citizens for FD?"
    
    # Call get_answer and print the result
    answer = get_answer(question)
    print("Answer:", answer)







