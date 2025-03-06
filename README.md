# AI Medical Chatbot

A powerful Medical Chatbot application built using **Streamlit**, **LangChain**, and **Pinecone**. This tool is designed to answer medical-related queries by utilizing user-provided datasets, such as PDFs, URLs, or predefined medical knowledge. By leveraging advanced machine learning and language processing techniques, the chatbot ensures precise and contextually relevant responses.

---

## Key Features

- **User-Friendly Chat Interface**: Engage in an interactive conversation to receive medical insights.
- **Multiple Data Input Methods**:
  - Upload PDFs containing medical knowledge.
  - Provide URLs to extract relevant medical content.
  - Utilize the default medical dataset focused on viruses.
- **Efficient Data Handling**: Automatically segments data into manageable chunks and stores them in a vector database for fast retrieval.
- **Custom Prompt Templates**: Ensures well-structured and contextually accurate responses.
- **Chat History**: Keeps track of previous conversations for future reference.

---



## Technologies Utilized

- **Streamlit**: Enables the development of an interactive web interface.
- **LangChain**: Supports intelligent retrieval-based question-answering.
- **Pinecone**: Serves as a high-performance vector database for optimized data search.
- **HuggingFace Embeddings**: Converts text into numerical vector representations for better analysis.
- **PyPDFLoader**: Facilitates the extraction and parsing of text from PDF files.

---
