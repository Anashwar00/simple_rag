import openai
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import dotenv
from dotenv import load_dotenv
import os
import pandas as pd
import uuid
import PyPDF2
import re
from PyPDF2 import PdfReader



load_dotenv()
openai.api_key="OPENAI_API_KEY"
class embedding_model:
    def __init__(self,model_type):
        if model_type=="openai":
            self.embd_model=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY")
                
            )
    def gen_model(self):
        return self.embd_model

class llm_type:
    def __init__(self,model):
        if model=="openai":
            self.client=OpenAI()
            self.model="gpt-4" 
            
    def generate_ans(self,message):
        try:
            res=self.client.chat.completions.create(
            model=self.model,
            messages=message,
            temperature=0.7
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"Eror:->{str(e)}")
            
def chunking():
    document = []
    try:
        with open("ml.pdf", "rb") as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            for i in range(num_pages):
                page = reader.pages[i]
                page_text = page.extract_text()
                if not page_text:
                    continue  # Skip pages with no text
                # Clean up text on a per-page basis
                cleaned_text = re.sub(r'[\x00-\x1F\x7F]+', '', page_text)
                new_text = re.sub(r"Download from finelybook.*?www\.finelybook\.com", "", cleaned_text)
                words = new_text.split()
                # Chunk the page's words into overlapping windows
                st, end = 0, 100
                while st < len(words):
                    chunk_words = words[st:end]
                    sec_text = " ".join(chunk_words)
                    document.append(sec_text.encode('utf-8', 'replace').decode('utf-8'))
                    st = end - 20  # overlap
                    end += 100
    except FileNotFoundError:
        print("The file 'ml.pdf' was not found.")
    return document
def database_setup(embd,document):
    client=chromadb.PersistentClient("./db/chroma")
    collection=client.get_or_create_collection(
        "ml_collection",
        embedding_function=embd
    )
    ids=[str(uuid.uuid4()) for _ in document]
    collection.add(
        ids=ids,
        documents=document
    )
    return collection


def user_query(query,collection):
    res=collection.query(
        query_texts=[query],
        n_results=10
    )
    return res

def augmentprompt(query, context):
    # Flatten any inner lists into a flat list of strings
    flat_docs = []
    for doc in context['documents']:
        if isinstance(doc, list):
            flat_docs.extend(doc)
        else:
            flat_docs.append(doc)
    context_str = "\n".join(flat_docs)
    prompt = f"""You are a very helpful assistant who helps people to answer ML related questions by using
context below-
context:{context_str}

query:{query}
"""
    return prompt

def rag_pipeline(query,collection,llm):
    chunk=user_query(query,collection)
    prompt=augmentprompt(query,chunk)
    message=[{"role":"system","content":"u are a helpfull assistent"},{
        "role":"user","content":prompt
    }]
    response=llm.generate_ans(message)
    return response




emb=embedding_model("openai")
embd=emb.gen_model()
llm=llm_type("openai")  
document=chunking()
collection=database_setup(embd,document)
while True:
    query=input("enter your query:-")
    if query in ["quit","exit"]:
        break
    response=rag_pipeline(query,collection,llm)
    print(response)
    
    



    






# response=
# collection=chromadb(embd,document)

    
    
    
    
    
    
            

        
        
        
        
    