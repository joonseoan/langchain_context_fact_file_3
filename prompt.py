# When the user ask questions, we need to run this file.

from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()

"""
                            [[[ Accept User Question and Answer the Question ]]]

"""

# chat model
chat = ChatOpenAI()

# Sets up the embedding tool
embeddings = OpenAIEmbeddings()

# Definitely, we need to access to ChromaDB whenever the user asks a question.
# For calculating embedding and finding similarity in ChromaDB
# we are not going to add documents (splitted chunks of texts) immediately here now.
db = Chroma(
  # using the same sqlite db
  persist_directory="emb",
  # LangChain is just a little bit disorganized
  # need to use keyword argument (**kargs) should be different, compared to the instance in main.py
  embedding_function=embeddings,
)

# We can manually use ChatPromptTemplate that receives SystemMessagePromptTemplate's fact from vector store
# and HumanMessagePromptTemplate's question from the user.

# However, LangChain already does it for us (--> RetrievalQA). It is a still chain under the hood.
# It is just got some different parts inside of it.

"""

retriever: is an object that has a method called `get_relevant_documents`
This method must take in a string and returns a list of documents. So if we have absolutely
any object in our application with a method called specifically `get_relevant_documents`, if that
method takes in a string, then gives us back a list of documents, we can refer to it as being a retriever.
Briefly, the user inputs a text by using `retriever`, and then it gives us a list of documents.

For instance, we use a particular vector database here called Chroma and this thing has a ton of very
specific functions tied to it. The functions like say `similarity_search`. We used it for testing. We
saw that if we put in some kind of string here and then we got back a list of documents tied to that string.
"""

# It is a interface between ChromaDB and RetrieverQA
# RetrieverQA deals with a ton of different vector database
# and then it needs an object which is a common object with a method taking string and returns documents,
# from many vector dbs. It delvers the object(db here) automatically with a method like `similarity_search`
# to RetrieverQA as `get_relevant_documents(string)` (Actually, `get_relevant_documents` calls `similarity_search(string)`)

"""
custom retriever to prevent the duplicated embeddings and their answers

class CustomFilterRetriever:
  def get_relevant_documents(self, query):
    # Code to use Chroma to find relevant docs (based on the question (query))
    # and remove any duplicate records

To make a custom retriever in general, we are going to create a class, it is going to extend
a base class of `BaseRetriever`. It has to have:
  1) `get_relevant_documents(self, query)` returns a list of documents
  2) async `aget_relevant_documents(self)` returns a list of documents
"""
retriever = db.as_retriever()

# `RetrievalQA` is able to takes in SystemMessagePromptTemplate and HumanMessageTemplate in chaining
# conversation and also access to Vector Store (DB). It is why we need to use RetrievalQA.
chain = RetrievalQA.from_chain_type(
  llm=chat,
  retriever=retriever,
  # We takes the relevant documents from vector store and then that document is placed 
  # at {fact} to be used to answer the user's question in SystemMessagePromptTemplate. 
  # (FYI, And the user's question is going to be at {question} in HumanMessagePromptTemplate)
  # So we are really just taking these documents out of the vector store and kind of shoving them into or
  # injecting them, or stuffing them, into the SystemMessageTemplate. That is what the term `chain_type="stuff"`
  chain_type="stuff"
)

result = chain.run("What is an interesting fact about the English language?")
# Use python prompt.py
print(result)
