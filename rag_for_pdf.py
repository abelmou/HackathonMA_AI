import ollama
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

pdf_doc = "data/World-Health-Organization.pdf"

if pdf_doc:
    loader = UnstructuredPDFLoader(file_path=pdf_doc)
    data = loader.load()
    print("PDF Loaded")
else:
    print("Upload a PDF File")

content = data[0].page_content
# print(content[:100])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("Splitting Done")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("Done Adding To Vector Database")

llm = ChatOllama(model="llama3.2")

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# We add answers of this Questions to Vector DB
# res = chain.invoke(input=("What is the document about?",))
# res = chain.invoke(input=("Who is the intended audience for this document?",))
# res = chain.invoke(input=("What questions should I ask the patient based on this document?",))
# res = chain.invoke(input=("What are the key medical terms or diagnoses mentioned in the document?",))
# res = chain.invoke(input=("What medications or treatments are prescribed, including their dosages?",))
# res = chain.invoke(input=("What are the main points as a healthcare assistant I should be aware of?",))
res = chain.invoke(input=("What steps should the healthcare assistant take based on the document's recommendations?",))

print(res)