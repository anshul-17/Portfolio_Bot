import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def process_pdf():
    loader = PyPDFLoader('CV.pdf')
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=250)
    chunks = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory="./chroma_db_v2")
    return vectordb


st.set_page_config(page_title="Anshul's AI Assistant",
                   page_icon=":robot_face:")
st.header("Chat with Anshul's Portfolio Bot")
st.write(
    "Ask me anything about Anshul's experience, skills, or projects based on his CV. I'm here to help you learn more about him!")


if "vector_db" not in st.session_state:
    with st.spinner("Analyzing resume data using local NLP embeddings..."):
        try:
            st.session_state.vector_db = process_pdf()
            st.success("CV Index successfully loaded!")
        except Exception as e:
            st.error(f"Failed to initialize Vector Database: {e}")
            st.stop()

if "vector_db" in st.session_state and st.session_state.vector_db is not None:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.2)

    system_prompt = (
        "You are a helpful, professional AI assistant representing Anshul Kumar, an AI Engineer. "
        "Review the entire context provided below containing his resume data to answer the user's question. "
        "Provide detailed explanations about his technical projects, tools used, and methodologies. "
        "If the specific detail isn't in the context, summarize what is available regarding his work experience. "
        "\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(
        st.session_state.vector_db.as_retriever(search_kwargs={"k": 5}),
        combine_docs_chain
    )
else:
    st.warning("Awaiting database initialization...")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = retrieval_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
