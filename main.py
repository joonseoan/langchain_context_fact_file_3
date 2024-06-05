from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
# [IMPORTANT]
# Need to install openai and tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


"""
                            [[[ Embeddings Setup to Chroma DB ]]]

"""

# 1) only load a file
# loader = TextLoader("facts.txt")
# docs = loader.load()

# print(docs)  # See the above comments

# LangChain provides classes to help load data from different types of files from the local machine
# We need to use a specific loader in terms of a file type
# Also, we are sometimes required to install additional packages to use each loader.
# For instance, in order to use PyPDFLoader, we need to additionally load PyPDF.

"""
txt : TextLoader use langchain.document_loaders
pdf: PyPDFLoader (need to install PyPDF)
json: JSONLoader use langchain.document_loaders
md: UnstructuredMarkdownLoader use langchain.document_loaders
"""

# LangChain also provides class that has ability to point to some remote location where
# some files are stored and load up them to the code.
# From the remote storage, we can use a single `Loader` for any kind of files (by default).
# For instance, `S3FileLoader` is able to load a ton of different kinds of files such as
# json, md, pdf, and txt to Amazon S3 (at once)

"""
Process installing a file
1. facts.txt
2. TextLoader take facts.txt
3. Result of loading a file is "Document"
---------------------------------------
            Document
  |page_content: file's content
  |metadata: {"source": "facts.txt"}
---------------------------------------
"""

"""
  1) If we insert entire text file into prompt: expensive run time
  2) If we insert only one text fact that matched with the user question:
    High possibility to find unrelated answer. We need to somehow find the most relative facts.
  3) If we insert one by one after splitting up the text by fact:
    It could generate the unrelated answer when the user question is obscure.

  Alternative - `Semantic Search`
  We are going to understand what the user is trying to get at, what question they are truly trying to ask.
  We are going to use `embeddings`

  -----------------------------------------------------------------------------------------------------------
  Sentence                              bravery score             happiness score               embeddings
                                        (1=bravery, -1=fear)      (1=happiness, -1=sad)
  -----------------------------------------------------------------------------------------------------------
  The happy child jumped bravery...            1                          1                       [1, 1]

  The child is not timid and had a
  good time....                               .5                         .3                       [.5, .3]

  Although filled with great fear,
  the child jumped from rock to rock          -1                          0 (nothing related)     [-1, 0]
  ---------------------------------------------------------------------------------------
  
  `embeddings` is a list of numbers between -1 and 1 that score how much a piece of text is taking about
  some particular quality.

  We can plot the embedding quality in a coordinate diagram (happy, bravery, fear, and sad)

                Happy 1
                  |                 *
                  |     *
  Fear -1 *-------------------- Bravery 1
                  |origin 0
                  |
                Sad -1

  Then we can draw the line from the origin to each value. Refer to each of these
  kinda lines right here as `vectors`. We can also refer to the embedding value themselves
  as being vectors.


  How can we sense each sentence is similar?
  1. the shorter distance each other, the more similar
    Squared L2 using the distance between two points to figure out how similar they are
                        +
  2. the more same angle each other, the more similar
    Cosine similarity using the angle between two vectors to figure out how similar they are

  ** even their quality of the questions are different.

  [The process]
  1. Loading the file
  2. Some `pre-processing` on this file before the user asks questions
    - split the loaded data into a bunch of separate chunks
    - calculate `embeddings` for each chunk
    - store each embeddings in a database specialized in storing embeddings (which is called Vector Store)
  3. Then wait for user's question
  4. When it takes the user's question, will create embeddings out of the user's question
  5. Do similar search with stored embeddings. We feed that embedding of the question into Vector Store. 
    At this moment, rather than store at this time, we are going to ask Vector Store to find maybe the 1, 2, 3 or 4 
    most similar vector that is stored or the most similar embeddings that it has stored.
  6. Then we are going to get out some very relevant chunk of text
  7. Then we put the chunk into Prompt.
"""

embeddings = OpenAIEmbeddings()

# For testing
# emb = embeddings.embed_query("hi there")
# print(len(emb))  # generates 1536 scores for the text "Hi there" from OpenAIEmbeddings

# split the file setup
# take a big blob of text and break it up into separate little chunks
text_splitter = CharacterTextSplitter(
  # tell the `text_splitter` what character we want to attempt to split our text on.
  # In our scenario, new character should be separator
  separator="\n",
  # the max is 200 characters long
  # even though we have a separator "\n", but we have chunk size 200.
  # therefore, a chunk can have 3 lines, for instance.
  chunk_size=200,
  # put some kind of copying of text between individual chunk
  # Sometimes the chunk looks awkward. So it puts the number of *last character*
  # to make sure that we do not kind of divide up everything into awkward chunks
  # It happens normally in PDF file
  chunk_overlap=0
)

# (1) loading the file
loader = TextLoader("facts.txt")

# (2) After load, it split the file.
# [IMPORTANT] `text_spitter` is required
docs = loader.load_and_split(
  text_splitter=text_splitter
)

# Basic docs format
# since we used splitter it has multiple document
# ex) [
#   Document(page_content='1. "Dreamt" is the only English....', metatdata={'surce': 'fact.txt'}),
#   Document(page_content='7. "The letter 'Q'....', metatdata={'surce': 'fact.txt'}),
# ]
# print(docs)  # See the above comments

# for doc in docs:
  # each Document in chunk
  # print(doc.page_content)
  # print("\n")


# (3) calculate embeddings for the loaded texts (using OpenAIEmbeddings here)
"""
There are different calculating models out there.
We will look at 2 models: `SentenceTransformer` and `OpenAI Embeddings`

The below `dimensions` word means that the list of numbers that scores a piece of text
on a ton of different qualities, that list of numbers is gonna have 768 elements

`SentenceTransformer` is used in the local machine. (Required processing power)
    It has several sub different models. These generate the different lengths
    For instance, `all-mpnet-base-v2` generates embeddings 768 dimensions.

`OpenAI Embeddings` is used in remote. It generates 1536 dimensions

Which one better? It depends on the usage of our application scenario.
Each embedding models is not compatible each other. (can't be used together)
"""

"""
  ChromaDB (open source)
  This is a vector store on our local machine. Internally, ChromaDB uses SQLite.
  It also includes a couple of libraries that are gonna do the actual math, all the actual computation
  to find embeddings similar to the ones that are stored inside of that SQLite database.

  pip install chromadb
  from langchain.vectorstores.chroma import Chroma
"""

# (4) store embeddings to Vector Store (will use ChromaDB)

# If we run the app over and over the multiple same records will be accumulated in vector store.
# We need to break our programs into two separate files. 
#  The first one is for the steps to store vector
#  The second one is for the steps for the users to ask question and show its results

# [Steps to create ChromaDB, a vector store]
# - Delete "emb" directory if it exists.
# - Run "python main.py" only one time. (creating nothing but uniq embeddings record)

"""
In LangChain, embeddings are used to represent text data in a numerical format that can be processed 
by machine learning models. 
Here's a more detailed explanation of how embeddings are handled in different stages:

[Storing Text Data]
Initial Calculation of Embeddings:
  When you store text data in a database or a vector store (such as Pinecone, Weaviate, or FAISS) 
  using LangChain, embeddings are calculated for each piece of text. 
  This involves using a pre-trained language model (like OpenAI's models, BERT, etc.) 
  to convert the text into a vector of numbers. 
  These vectors are then stored in the database along with the original text.

[Accepting a User Query]
Recalculation of Embeddings:
When a user submits a query, LangChain recalculates the embeddings for the query text.
This step is necessary because the system needs to convert the query into 
the same numerical format (vector) as the stored text
to perform `similarity searches` or other operations.

[Workflow]
Text Storage:

1. Text data is converted to embeddings.
2. Embeddings and text data are stored in the database/vector store.

Handling a User Query:

1. The user's query is received.
2. The query is converted into an embedding using the same model that was used for the original text data.
3. The query embedding is then compared with the stored embeddings to find the most similar texts 
  or perform other relevant operations.

[Why Recalculate?]
Recalculating embeddings for the user query is essential because:

- Consistency: The query needs to be in the same vector space as the stored data for meaningful comparison.
- Dynamic Queries: User queries are dynamic and can vary each time, requiring fresh embeddings 
for accurate results.

[Example Process]
Storing Text:
1. Text: "LangChain is a framework for developing applications using language models."
2. Embedding: [0.23, -0.15, 0.45, ...] ***** (calculated and stored) *****

User Query:
1. Query: "What is LangChain?"
2. Query Embedding: [0.25, -0.14, 0.46, ...] ***** (calculated at query time) *****
3. Similarity Search: Compare query embedding with stored embeddings to find relevant text.

[Efficiency Considerations]
Batch Processing: For large datasets, embeddings can be calculated in batches to improve efficiency.
Caching: Some systems might cache frequently asked queries and their embeddings to speed up response times.
Understanding this process is crucial for effectively using LangChain and similar frameworks that rely on embeddings for text processing and retrieval.

"""

# Setting ChromaDB instance
db = Chroma.from_documents(
  # importing docs for `embeddings`
  docs,
  # calculate embeddings using openAIEmbedding solution for each chunk of the texts.
  # Please notice `embedding`, not `embeddings` for the key argument.
  embedding=embeddings,
  # After calculating embeddings, they are going to be placed inside of "" directory
  persist_directory="emb",
)

# (5) [Steps for the user to ask questions and for resulting in the answer]
# - Create a file "prompt.py". This is the file we are going to run anytime we want to ask some question
#   of ChatGPT and use some content of our vector database to provide some context.
# - Build "prompt.py"

"""
  How to prevent the same embeddings to be saved in ChromaDB
  whenever run this main.py?

  Use `EmbeddingsRedundantFilter`.
  Taking some documents (like fact.txt) and passing them into this class
  And this class calculates embeddings and then it is going to compare all these embeddings
  against each other. If any of them are similar, then it will be removed.

  The downside of `EmbeddingsRedundantFilter` is that
  because it only works against the currently loading documents,

  [at the input step]
  when the embeddings are already are installed and used,
  there is no way to recalculate new embeddings against the stored embeddings

  [at the output step]
  Also, we can't easily insert or inject `EmbeddingsRedundantFilter` into
  `RetrievalQA Chain to Chroma Retriever` in prompt.py which could be another chance for us
  to filter out the similar chunk of the answer.

  We will implement the custom retriever at the output step in prompt_using_redundant.py as an workaround.
  Please, find prompt.py.
"""

# For testing only. `prompt.py` will ask question instead. ----------------------------------------------
# "ask the question" where what we want to find some documents stored inside of a DB related to.


# 2)
# set tuples for similarity score and chunks
# results = db.similarity_search_with_score(
#   "What is an interesting fact about the English language?",
#   # the number of result on the basis of the highest similarity
#   # k=1, 2, 3
# )

# 1) only for chunk
# results = db.similarity_search(
#   "What is an interesting fact about the English language?",
#   # k=1
# )

# [IMPORTANT]
# It shows a duplicated answers
# because main.py runs a several times
# Therefore, we need to implement `EmbeddingsRedundantFilter`
# Please find `prompt.using_redundant.py`
# for result in results:
#   print("\n")

  # 2) set tuples for similarity score and chunks
  # It shows chunk by chunk, not line by line with similarity score

  # It is a tuple.
  # This is the similarity score ("0.3502454302737522")
  # print(result[1])
  # print(result[0].page_content)

  # 1) It show chunk only
  # print(result.page_content)
# ------------------------------------------------------------------------------------------------------
