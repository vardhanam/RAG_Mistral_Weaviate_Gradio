from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
BitsAndBytesConfig,
pipeline
)

import torch

import os

from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

from operator import itemgetter

import gradio as gr

import weaviate



def load_llm():

    #Loading the Mistral Model
    model_name='mistralai/Mistral-7B-Instruct-v0.2'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )

    # Building a LLM text-generation pipeline
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1024,
        device_map = 'auto',
    )


    return text_generation_pipeline

def embeddings_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return embeddings


def initialize_vectorstore():

    vectorstore = Weaviate.from_documents(
        [], embedding=hf_embeddings,
        weaviate_url = 'https://superteams-810p8edk.weaviate.network',
        by_text= False
    )

    return vectorstore

def text_splitter():
    # Simulate some document processing delay
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter

def add_pdfs_to_vectorstore(files):

    saved_files_count = 0
    for file_path in files:
        file_name = os.path.basename(file_path)  # Extract the filename from the full path
        if file_name.lower().endswith('.pdf'):  # Check if the file is a PDF
            saved_files_count += 1
            loader_temp = PyPDFLoader(file_path)
            docs_temp = loader_temp.load_and_split(text_splitter=textsplitter)
            for doc in docs_temp:
                # Replace all occurrences of '\n' with a space ' '
                doc.page_content = doc.page_content.replace('\n', ' ')
            vectorstore.add_documents(docs_temp)

        else:
            print(f"Skipping non-PDF file: {file_name}")

    return f"Added {saved_files_count} PDF file(s) to vectorstore/"

def weaviate_client():
    client = weaviate.Client(url=  'https://superteams-810p8edk.weaviate.network')
    return client


def answer_query(message, chat_history):
    context_docs = vectorstore.similarity_search(message, k= 3)
    context = ' '.join(doc.page_content for doc in context_docs)

    template = f"""Answer the question based only on the following context:
        {context}

        Question: {message}
    """

    result = llm(template)

    answer = result[0]["generated_text"].replace(template, '')

    chat_history.append((message, answer))

    return "", chat_history

def clear_vectordb(chatbot, msg):
    client.schema.delete_all()
    chatbot = ""
    msg = ""
    return chatbot, msg

llm = load_llm()

hf_embeddings = embeddings_model()

client = weaviate_client()

vectorstore = initialize_vectorstore()

textsplitter = text_splitter()


with gr.Blocks() as demo:
    with gr.Row():
        upload_files = gr.File(label="Upload pdf files only", file_count='multiple')
        success_msg = gr.Text(value="")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your query here")
    clear = gr.Button("Clear VectorDB and Chat")

    upload_files.upload(add_pdfs_to_vectorstore, upload_files, success_msg)
    msg.submit(answer_query, [msg, chatbot], [msg, chatbot])
    clear.click(clear_vectordb, [chatbot, msg], [chatbot, msg])

demo.launch(server_name='0.0.0.0', share= True)






