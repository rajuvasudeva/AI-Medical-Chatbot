import tempfile
import streamlit as st
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as pc, ServerlessSpec
import os
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from functions import data_load_url,data_load_pdf,text_split,huggingface_embeddings

load_dotenv(dotenv_path='.env')
GROQ_API = os.getenv('GROQ_API')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


st.set_page_config(layout="wide", page_title="Medical Chatbot", page_icon="🤖")
st.sidebar.header(":blue[Pick a Data Source for Analysis]")
data_type = st.sidebar.selectbox("Select Input Type:", ("Default", "PDF", 'URL'))

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def training(type, data=None, chunk_size=500, chunk_overlap=50):
    for key in st.session_state.keys():
        if key != "chat_history":
            del st.session_state[key]
    index_name=''
    embeddings = huggingface_embeddings()
    vector = embeddings.embed_query('helloo')
    dimensionss = len(vector)
    pine = pc(api_key=PINECONE_API_KEY)
    if type == 'PDF' or type == 'URL':
        text_chunks = text_split(data, chunk_size, chunk_overlap)
        if 'pdfurl' in pine.list_indexes().names():
            pine.delete_index(name='pdfurl')
        index_name = 'pdfurl'
        cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
        region = os.environ.get('PINECONE_REGION') or 'us-east-1'
        spec = ServerlessSpec(cloud=cloud, region=region)
        if index_name not in pine.list_indexes().names():
            pine.create_index(
                name=index_name,
                dimension=dimensionss,
                metric="cosine",
                spec=spec
            )
            while not pine.describe_index(index_name).status['ready']:
                time.sleep(1)

        vectorstore_from_docs = PineconeVectorStore.from_documents(text_chunks, index_name=index_name,
                                                                   embedding=embeddings)
    else:
        index_name = 'data'
        if index_name not in pine.list_indexes().names():
            text_chunks = text_split(data, chunk_size, chunk_overlap)
            cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
            region = os.environ.get('PINECONE_REGION') or 'us-east-1'
            spec = ServerlessSpec(cloud=cloud, region=region)
            if index_name not in pine.list_indexes().names():
                pine.create_index(
                    name=index_name,
                    dimension=dimensionss,
                    metric="cosine",
                    spec=spec
                )
                while not pine.describe_index(index_name).status['ready']:
                    time.sleep(1)

            vectorstore_from_docs = PineconeVectorStore.from_documents(text_chunks, index_name=index_name,
                                                                       embedding=embeddings)
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
    loaded_msg=st.success("Index Loaded!!")
    time.sleep(3)
    loaded_msg.empty()
    prompt_template = """Use the GIVEN INFORMATION CONTEXT ONLY to give appropriate answer for the user's question.
                If you don't know the answer , just say I DON'T KNOW THE ANSWER or QUESTION IS OUT-OF-CONTEXT but Don't make up an answer.
                Context: {context}
                Question: {question}
                Only return the appropriate answer and nothing else.
                Helpful answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGroq(model_name="llama-3.2-3b-preview", api_key=GROQ_API, temperature=0.3)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=docsearch.as_retriever(search_kwargs={"k": 12}),
                                     return_source_documents=True, chain_type_kwargs={"prompt": prompt})
    analysis = st.success("Analysis Completed! Your Medical Chatbot is Ready!✅")
    time.sleep(3)
    analysis.empty()
    st.session_state['qa'] = qa



data = []
if data_type == "PDF":
    file = st.sidebar.file_uploader("Upload your PDF", type="pdf")


    if file:
        submitted = st.sidebar.button("Submit")
        if submitted:
            with st.spinner("Processing your data..."):
                data = data_load_pdf(file)
                training(data=data, type=data_type)

if data_type == "URL":
    st.sidebar.info("For Entering Multiple URLs, Give Single space between URLs")
    url = st.sidebar.text_area("Enter URL")
    urls = url.split(' ')
    if urls and st.sidebar.button("Submit"):
        with st.spinner("Processing your data..."):
            data = data_load_url(urls)
            training(data=data, type=data_type)
if data_type == "Default":
    st.sidebar.info("You are using Default Data")
    if st.sidebar.button("Submit"):
        with st.spinner("Processing your data..."):
            loader = PyPDFLoader('Medical_PDF.pdf')
            data = loader.load()
            training('Default', data=data,chunk_size=1500,chunk_overlap= 150)
if data_type:
    st.title(":blue[Medical ChatBot] :robot_face:")
    st.subheader("Ask Your Questions")
    inputt = st.text_input("Type your Question Here")
    warn = st.warning("Please Provide Data for Analysis")
    if 'qa' in st.session_state:
        warn.empty()
        if st.button("Get Answer"):
            if inputt:
                with st.spinner("Generating your answer..."):
                    qa = st.session_state['qa']
                    result = qa.invoke(inputt)

                    st.success("Here is the answer to your question.")
                    st.write(result['result'])
                    st.session_state['chat_history'].append({"question": inputt, "answer": result['result']})

            else:
                st.warning("Enter a question")

    if 'chat_history'  in st.session_state:
        st.markdown("## :blue[ Chat History]")
        with st.expander("View Entire Chat History", expanded=True):
            for i, entry in enumerate(st.session_state['chat_history']):
                st.write(f"# Query {i + 1}")
                st.write(f":red[**Question:**] {entry['question']}")
                st.write(f":red[**Answer:**] {entry['answer']}")
                st.divider()




