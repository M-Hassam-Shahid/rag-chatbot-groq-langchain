import streamlit as st

# phase 2
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# phase 3 - RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


st.title("RAG CHATBOT!")


# Setup a session chat variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


# Create vector database from PDF
@st.cache_resource
def get_vectorstore():

    # Load PDF
    loader = PyPDFLoader("ai_ml_rag_sample.pdf")
    documents = loader.load()

    # Split PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


prompt = st.chat_input("Pass your prompt here!")


if prompt:

    # Display user message
    st.chat_message("user").markdown(prompt)

    # Save user message
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })


    try:

        # Get vector store
        vectorstore = get_vectorstore()


        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )


        # Search PDF for relevant information
        relevant_documents = retriever.invoke(prompt)


        # Combine relevant chunks
        context = "\n\n".join(
            document.page_content
            for document in relevant_documents
        )


        # RAG prompt
        groq_sys_prompt = ChatPromptTemplate.from_template("""
You are a helpful RAG assistant.

Answer the following question using ONLY the provided context.

If the answer is not available in the context,
say "I don't know based on the provided document."

Do not make up information.

Context:
{context}

Question:
{user_prompt}

Start the answer directly. No small talk please.
""")


        # Groq model
        model = "openai/gpt-oss-120b"

        groq_chat = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            model_name=model
        )


        # Create chain
        chain = groq_sys_prompt | groq_chat | StrOutputParser()


        # Send question + PDF context to model
        response = chain.invoke({
            "user_prompt": prompt,
            "context": context
        })


        # Display answer
        st.chat_message("assistant").markdown(response)


        # Save answer in chat history
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })


    except Exception as e:
        st.error(f"Error: {str(e)}")

