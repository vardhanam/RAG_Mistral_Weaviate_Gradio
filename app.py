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
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter



def return_chain_elements():

    #template to get the Standalone question
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    #Function to create the context from retrieved documents
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    #Creating the template for the final answer
    template = """Answer the question based only on the following context:
        {context}

        Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)


    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    }

    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | vectorstore.as_retriever(search_kwargs = {'k':10}),
        "question": lambda x: x["standalone_question"],
    }

    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
        "docs": itemgetter("docs"),
    }

    return standalone_question, retrieved_documents, answer


def add_pdfs_to_vectorstore(files):

    saved_files_count = 0
    for file_path in files:
        file_name = os.path.basename(file_path)  # Extract the filename from the full path
        if file_name.lower().endswith('.pdf'):  # Check if the file is a PDF
            saved_files_count += 1
            loader_temp = PyPDFLoader(file_path)
            docs_temp = loader_temp.load_and_split(text_splitter=textsplitter)
            vectorstore.add_documents(docs_temp)

        else:
            print(f"Skipping non-PDF file: {file_name}")

    return f"Added {saved_files_count} PDF file(s) to vectorstore/"

def weaviate_client():
    client = weaviate.Client(url=  'https://superteams-810p8edk.weaviate.network')
    return client


def answer_query(message, chat_history):
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(conversational_memory.load_memory_variables) | itemgetter("history"),
    )
    chain = loaded_memory | standalone_question | retrieved_documents | answer

    inputs = {"question": message}
    result = chain.invoke(inputs)

    chat_history.append((message, result["answer"]))
    conversational_memory.save_context(inputs, {"answer": result["answer"]})

    return "", chat_history

def clear_vectordb(chatbot, msg):
    client.schema.delete_all()
    hatbot = ""
    msg = ""
    return chatbot, msg

llm = load_llm()

hf_embeddings = embeddings_model()

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






