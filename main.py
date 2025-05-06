from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Load and split the document
loader = PyPDFLoader("data/Alice_In_Worderland.pdf")  # Replace with your actual PDF filename
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
texts = text_splitter.split_documents(data)

# Create vector store
vector_store = FAISS.from_documents(
    texts,
    VertexAIEmbeddings(model="text-embedding-004")
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")  # Updated model name format

# Create retrieval chain
retriever = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

while True:
    query = input()
    if query in ["exit","-1","quit"]:
        break
    response = retrieval_chain.invoke({"input": query})
    print(response["answer"])