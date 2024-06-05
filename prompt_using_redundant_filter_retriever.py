from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

# only for debug to see if the duplicate records and / or answers exist
# since we used `RedundantFilterRetriever`
langchain.debug = True

load_dotenv()

"""
                            [[[ Accept User Question and Answer the Question ]]]

"""

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings,
)

"""
custom retriever to prevent the duplicated embeddings and their answers

class RedundantFilterRetriever:
  def get_relevant_documents(self, query):
    # Code to use Chroma to find relevant docs (based on the question (query))
    # and remove any duplicate records


To make a custom retriever in general, we are going to create a class, it is going to extend
a base class of `BaseRetriever`. It has to have:
  1) `get_relevant_documents(self, query)` returns a list of documents
  2) async `aget_relevant_documents(self)` returns a list of documents

test: 1) run python main.py and add the same embedding several times
      2) set debug mode like the one above
      3) run prompt_using_redundant.....py
      4) we can't find the duplicate answers in `context` property below
"""
"""
[chain/start] [chain:RetrievalQA > chain:StuffDocumentsChain > chain:LLMChain] Entering Chain run with input:
{
  "question": "What is an interesting fact about the English language?",
  "context": "1. \"Dreamt\" is the only English word that ends with the letters \"mt.\"
            \n2. An ostrich's eye is bigger than its brain.
            \n3. Honey is the only natural food that is made without destroying any kind of life.
            
          \n\n4. A snail can sleep for three years.
            \n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'
            \n6. The elephant is the only mammal that can't jump.
            
          \n\n86. Broccoli and cauliflower are the only vegetables that are flowers.
            \n87. The dot over an 'i' or 'j' is called a tittle.
            \n88. A group of owls is called a parliament.
              
          \n\n118. The original Star-Spangled Banner was sewn in Baltimore.
          \n119. The average adult spends more time on the toilet than they do exercising."
}

"""

# 2)
retriever = RedundantFilterRetriever(
  embeddings=embeddings,
  chroma=db,
)

# 1)
# For testing to see if the duplicate context chunks are available (should have them) it has
# retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
  llm=chat,
  retriever=retriever,
  chain_type="stuff"
)

result = chain.run("What is an interesting fact about the English language?")
print(result)


