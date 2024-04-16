# Import required libraries
from itertools import zip_longest
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from streamlit_chat import message
from langchain_openai import OpenAIEmbeddings
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
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader

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

# Initialize the ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo",
    api_key=st.secrets['openai_api_key']
)
# Intialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['openai_api_key'])

#Fetch and load documents
#path = "app/static/CV.pdf"
#loader = PyPDFLoader(path)
#docs = loader.load()

#Indexing
text_splitter = RecursiveCharacterTextSplitter()
#documents = text_splitter.split_documents(docs)
documents = [Document(page_content='Harald Andreas Osan, konsulent\nInger Bang Lunds vei 9, 5059, Bergen, Norge, 47679043, haraldosan@hotmail.com\nFødselsdato 13.11.1995\nFødested KristiansandNasjonalitet Norsk\nFørerkort Klasse B\nL I N K LinkedIn\nP R O F I L Engasjert konsulent med interesse for smidige løsninger, sømløse integrasjoner og forretninganalyse. Jeg arbeider \nmed skyløsninger, primært innen utvikling, dataarkitektur, analyse og datavitenskap. Har lidenskap for innsikt \ngenerert av ustrukturert data og er motivert av å finne de røde trådene i store datasett. Er også en ivrig podcastlytter \nog hobbykokk!\nA R B E I D S H I S TO R I K K \nOkt 2023 — Dags dato Senior Associate, Technology Consulting, KPMG Bergen\n•Ansvarlig for datamigrering i større konsern ved plattformbytte. Gjennomført kartlegging av nødvendige \ndataområder, satt opp rammeverk for gjennomføring av datamigrering, gjennomført datatransformasjon, \ndatavask i henhold til forretningsmessige behov, innlasting og feilhåndtering. Utviklet løsningsarkitektur \nfor tilgjengeliggjøring av data for analyse i Power BI via datavarehusløsning.\xa0\n•Utviklet løsningsarkitektur for innkjøpsprediksjon og lagerstyring i Microsoft Fabric for mindre \nhavbruksselskap.\n•Gjennomført flere ende til ende Cash to Revenue-analyser ved bruk av Power BI, Python og PowerQuery.\nAug 2021 — Sep 2023 Associate, Technology Consulting, KPMG Bergen\n•Implementering av Dynamics 365 Finance and Operations for stort oppdrettskonsern. Avansert bruk av \nkreditt-, forsikrings- og factoringsløsninger mot norsk finansinstitusjon. Delaktig i utvikling, testing og \nutrulling av løsning. Hovedansvar for vedlikehold og utviklet tilpasning for konsernkonsolidering.\xa0\n•Implementering av Dynamics 365 Finance and Operations for stort retailkonsern. Automatisering \nav innkjøpsprosess og leverandørreskontro. Utviklet maskinlæringspipelines for innkjøpspredisjon og \nlagerstyring med Azure MLS og Azure Data Lake Storage Gen 2.\n•Implementering av Dynamics 365 Finance and Operations for norsk forlag. Arbeid med dataflyt i Azure \nData Lake og Azure Synapse mot Amazon S3 for konsernrapportering.\nAug 2020 — Jun 2021 Seminarleder, Universitetet i Bergen Bergen\nSeminarleder i INFO-132, innføring i programmering i Python. Underviste to grupper med studenter og utførte \nsensur på innleveringer.\nJun 2020 — Aug 2020 Softwareutvikler, Universitetet i Bergen\nSommerjobb som utvikler. Benyttet teknologier som Python, Javascript, Svelte, Docker, Kubernetes og \nMongoDB. Utviklet plattform for testing av studenters programmeringsferdigheter\nU T DA N N I N G \nAug 2019 — Jun 2021 Bachelor i Informasjonsvitenskap, Universitetet i Bergen Bergen\nAug 2016 — Jun 2019 Bachelor i Økonomi og Administrasjon, Universitetet i Agder Kristiansand\nBacheloroppgave: Prissetting på kraftkontrakter i det kortsiktige terminmarkedet.\nF E R D I G H E T E R Python\nMicrosoft Excel\nPower BI\nMicrosoft FabricCloud Scale Analytics\nApache Spark v/ PySpark\nAzure Synapse\nMaskinlæring', metadata={'source': 'CV.pdf', 'page': 0}), Document(page_content='S P R Å K Norsk MorsmålEngelsk V eldig god kunnskap\nR E F E R A N S E R Referanser tilgjengelig ved forespørsel', metadata={'source':'CV.pdf', 'page': 1})]
vector = FAISS.from_documents(documents, embeddings)

#Set retriever and create retrieval chain
retriever = vector.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(chat, retriever, prompt)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

document_chain = create_stuff_documents_chain(chat, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content="You are a helpful AI assistant talking with a human. If you do not know an answer, just say 'I don't know', do not make up an answer.")]

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


# Add credit
st.markdown("""
---
Laget med 🤖 av Harald Osan""")
