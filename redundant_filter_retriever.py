from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

# BaseRetriever has some functions and attributes in it
# to kind of define what retriever is in general.
class RedundantFilterRetriever(BaseRetriever):
  embeddings: Embeddings
  chroma: Chroma

  # overriding the method in super class, BaseRetriever
  # `query` is a question string
  def get_relevant_documents(self, query: str):
    # 1) calculate embeddings for the `query` string on the basis AI libs (In our case, OpenAIEmbeddings)
    # `embed_query`: we can pass a string into it. It is going to use the OpenAI API
    # to calculate some embeddings for that string. (we did it before in prompt.py)

    # [type]
    # In our case, we are using OpenAIEmbeddings() but we should not define here
    # because we could use different libs. To effectively code for the different embeddings lib
    # we can use `Embeddings` type.
    emb = self.embeddings.embed_query(query)

    """
      (We did it before)
      embeddings = OpenAIEmbeddings()
      db = Chroma(persist_directory="emb", embedding_function=embeddings)

      result = db.similarity_search("What is an interesting face about...")
    """

    """
      In our case,
      embeddings = OpenAIEmbeddings()
      db = Chroma(persist_directory="emb", embedding_function=embeddings)

      emb = embeddings.embed_query("What is an interesting face about...")
      
      [IMPORTANT]
      # if we want to find similarities to an embedding we already calculated,
      # we can use the `similarity_search_by_vector` function instead of `similarity_search`
      
      # We just took a look at a way that we can find similar documents
      # when we have an existing string. But!!, we can also find related documents
      # to an embedding (stored record in DB). This is just kind of removing a step out of Chroma.
      # When we call `db.similarity_search(question)`, Chroma is going to calculate the embeddings for that string
      # and then do the similarity search.

      # If it turns out that we happen to already have the embeddings handy like ("emb = embeddings.embed_query("What is an interesting face about...")"), 
      # so we already calculated the embeddings ahead of time, we can do an identical similarity search but
      # base it upon a vector or an embedding, So all this is saying, "Hey, Chroma, do not worry about
      # calculating the embedding I already did it. You do not need to calculate embeddings. Only do
      # similarity search!

      # 1)
      results = db.similarity_search_by_vector(emb)

      # 2) same thing as the one above
      # Additionally, it can remove the duplicated stuff
      results = db.max_marginal_relevance_search_by_vector(
        embedding=emb,
        lambda_mult=0.8
      )
    """

    # take those embeddings and feed them into that
    # `max_marginal_relevance_search_by_vector`.
    # `max_marginal_relevance_search_by_vector` returns the list of documents
    # requirements: Chroma instance
    return self.chroma.max_marginal_relevance_search_by_vector(
      embedding=emb,
      # lambda_mult is a range from 0 to 1.
      # The higher number, the more generous to similarity
      lambda_mult=0.8,
    );

  # We are not going to worry about the async implementation
  async def aget_relevant_documents(self):
    return [];