# Import required libraries
from itertools import zip_longest
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from streamlit_chat import message
from langchain_openai import OpenAIEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # type: ignore
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

#Load environment variables
# Set streamlit page configuration
st.set_page_config(page_title="AI-Kurs ChatBot")
st.title("AI-Kurs ChatBot")

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.1 #Store user input for AI bot temperature   

# Initialize the ChatOpenAI model

def initialize_openai_bot():
    return ChatOpenAI(
        temperature=st.session_state['temperature'],
        model_name="gpt-4-turbo",
        api_key=st.secrets['openai_api_key']
    )
# Intialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['openai_api_key'])

#Fetch and load documents

def build_retriever():
    path = "https://if.no/apps/vilkarsbasendokument/Vilkaar?produkt=Generelle_vilk%C3%A5r"
    loader = PyPDFLoader(path)
    docs = loader.load()

    #Indexing
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    #documents = 
    vector = FAISS.from_documents(documents, embeddings)

    #Set retriever and create retrieval chain
    retriever = vector.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Gitt den tidligere samtalen, generer et søk som gir mening i relasjon til tidligere samtale")
    ])
    retriever_chain = create_history_aware_retriever(chat, retriever, prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Svar på brukerens spørsmål som en ekspert innen forsikring og vilkår:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(chat, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    return retrieval_chain

def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content="Du er en hjelpsom ekspert innen forsikring som hjelper kunder med spørsmål de har rundt generelle forsikringsvilkår. Dersom du ikke vet svaret kan du henvise dem til en kundebehandler.")]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages


def generate_response(user_input):
    """
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages = build_message_list()
    output = retrieval_chain.invoke({"chat_history":zipped_messages,"input":user_input})
    # Generate response using the chat model
    return output['answer']


# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""
    
def change_temp():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.temperature = st.session_state.prompt_temp
    chat = initialize_openai_bot()

# Create a text input for user
st.text_input('YOU: ', key='prompt_input', on_change=submit)


if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Generate response
    output = generate_response(user_query)

    # Append user query to past queries
    st.session_state.past.append(user_query)

    # Append AI response to generated responses
    st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i))
        # Display user message
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')

with st.sidebar:
    pdf_url = st.text_input(default=path,label="Legg til URL til PDF")
    st.slider(min_value=0,max_value=1,label="Juster temperaturen til AI",step=0.1,key="prompt_temp", on_change=change_temp)

chat = initialize_openai_bot()
retrieval_chain = build_retriever()

# Add credit
st.markdown("""
---
Laget med 🤖 av Harald Osan""")
