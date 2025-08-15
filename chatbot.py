import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.chat_models import ChatHuggingFace

# Use a small, ungated model (runs on CPU)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7
)


OPENAI_API_KEY = ""

#upload pdf files
st.header('My first Chatbot')

with st.sidebar:
    st.title('Your Documents')
    file = st.file_uploader("Upload a pdf file and start asking questions", type='pdf')


#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text=''
    for page in pdf_reader.pages:
        text+=page.extract_text()
        # st.write(text)



#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # generating embeddings
    # embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    # creating vectore store - FAISS
    # embeddings, initializ FAISS, store chunks and embeddings
    # vector_store = FAISS.from_texts(chunks,embeddings)

    #get user question
    user_question = st.text_input("Type your question here")

    #do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)

        # define the LLM
        # llm = ChatOpenAI(
        #     openai_api_key = OPENAI_API_KEY,
        #     temperature = 0,
        #     max_tokens = 1000,
        #     model_name = 'gpt-5'
        # )
        # llm = ChatOllama(model="llama3")  # or "mistral", "gemma"
        # llm = ChatHuggingFace(pipeline=pipe)  # ✅ works like ChatOpenAI
        llm = ChatHuggingFace(pipeline=pipe)


        #output
        chain = load_qa_chain(llm,chain_type='stuff')
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
