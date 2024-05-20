# # chat_models for conversational model
# from langchain.chat_models import ChatOpenAI
# # To avoid a deprecation warning, we need to update our import
# from langchain.chains import LLMChain
# from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
# from langchain.memory import ConversationSummaryMemory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
# [IMPORTANT]
# Need to install openai and tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# 1) only load a file
# loader = TextLoader("facts.txt")
# docs = loader.load()

# print(docs)  # See the above comments


# LangChain provides classes to help load data from different types of files from the local machine
# We need to use a specific loader in terms of a file type
# Also, we are sometimes required to install additional packages to use each loader.
# For instance, in order to use PyPDFLoader, we need to additionally load PyPDF.
"""
txt : TextLoader
pdf: PyPDFLoader
json: JSONLoader
md: Unstructured MarkdownLoader
"""

# LangChain also provides class that has ability to point to some remote location where
# some files are stored and load up them to the code. From the remote storage,
# we can use a single Loader for any kind of files (by default).
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


  Alternative - Semantic Search
  We are going to understand what the user is trying to get at, what question they are truly trying to ask.
  We are going to use `embeddings`

  -----------------------------------------------------------------------------------------------------------
  Sentence                              bravery score             happiness score               embeddings
                                        (1=bravery, -1=fear)      (1=happiness, -1=sad)
  -----------------------------------------------------------------------------------------------------------
  The happy child jumped bravery...            1                          1                       [1, 1]

  The child is not timid and had a
  good time....                               .5                         .3                       [.5, .3]

  Although filled wi great fear,
  the child jumped from rock to rock          -1                          0 (nothing related)     [-1, 0]
  ---------------------------------------------------------------------------------------
  
  `embeddings` is a list of numbers between -1 and 1 that score how much a piece of text is taking about
  some particular quality.

  We can plot the embedding quality in a coordinate diagram (happy, bravery, fear, and sad)

                Happy 1
                  |                 *
                  |     *
  Fear -1 *----------------------- Bravery 1
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
    Rather than store at this time, we are going to ask Vector Store to find maybe the 1, 2, 3 or 4 
    most similar vector that is stored or the most similar embeddings that it has stored.
  6. Then we are going to get out some very relevant chunk of text
  7. Then we put the chunk into Prompt.

"""

embeddings = OpenAIEmbeddings()

# For testing
# emb = embeddings.embed_query("hi there")
# print(len(emb))  # generates 1536 scores for the text "Hi there" from OpenAIEmbeddings

# split the file setup
# take a big  blob of text and break it up into separate little chunks
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
  # It happens normally inn PDF file
  chunk_overlap=0
)

# 1. loading the file
loader = TextLoader("facts.txt")

# 2) After load, it split the file.
# [IMPORTANT] `text_spitter` is required
docs = loader.load_and_split(
  text_splitter=text_splitter
)

for doc in docs:
  # each Document in chunk
  print(doc.page_content)
  print("\n")

# 3) calculate embeddings for the loaded texts (will use it after storing embedding because it is expensive)
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

# Store embeddings to Vector Store (will use ChromaDB)
"""
  ChromaDB (open source)
  This is a vector store on our local machine. Internally, ChromaDB uses SQLite.
  It also includes a couple of libraries that are gonna do the actual math, all the actual computation
  to find embeddings similar to the ones that are stored inside of that SQLite database.

  pip install chromadb
  from langchain.vectorstores.chroma import Chroma
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

# "ask the question" where what we want to find some documents stored inside of a DB related to.

# 2)
# set tuples for similarity score and chunks
# results = db.similarity_search_with_score(
#   "What is an interesting fact about the English language?",
#   # the number of result on the basis of the highest similarity
#   # k=1, 2, 3
# )

# 1) only for chunk
results = db.similarity_search(
  "What is an interesting fact about the English language?",
  k=1
)

print("")
print("------------------ Result -------------------")
for result in results:
  print("\n")

  # 2)
  # It shows chunk by chunk, not line by line with similarity score

  # It is a tuple.
  # This is the similarity score ("0.3502454302737522")
  # print(result[1])
  # print(result[0].page_content)

  # 1) It show chunk only
  print(result.page_content)


# Tomorrow: avoid storing the same data over and over to the DB.

# since we used splitter it has multiple document
# ex) [Document(question contents with "\n")]
# print(docs)  # See the above comments









