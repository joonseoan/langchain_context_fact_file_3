# chat_models for conversational model
from langchain.chat_models import ChatOpenAI
# To avoid a deprecation warning, we need to update our import
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()
