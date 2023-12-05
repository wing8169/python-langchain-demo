import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Sample usage
# text = "What would be a good company name for a company that makes colorful socks?"
# messages = [HumanMessage(content=text)]
# print(llm.invoke(text))

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)
# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
# print(conversation({"question": "hi"}))

# print(
#     chat_prompt.format_messages(
#         input_language="English", output_language="French", text="I love programming."
#     )
# )

# In actual usage, you would set `k` to be a higher value, but we use k=1 to show that
# the vector lookup still returns the semantically relevant information
embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=embedding_function,
    collection_name="user1",
)

retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# When added to an agent, the memory object can save pertinent information from conversations or used tools
# memory.save_context(
#     {"input": "My favorite food is pizza"}, {"output": "that's good to know"}
# )
# memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
# memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"})  #

# print(memory.load_memory_variables({"prompt": "what sport should i watch?"})["history"])

chain = prompt | ChatOpenAI() | StrOutputParser()

history = ChatMessageHistory()
historyDatas = memory.load_memory_variables({"prompt": "what sport should i watch?"})[
    "history"
]
print(historyDatas)
splitted = historyDatas.split("\n")
for i in range(len(splitted)):
    if i % 2 == 0:
        history.add_user_message(splitted[i])
    else:
        history.add_ai_message(splitted[i])
print(history)
# history.add_user_message("hi")
# history.add_ai_message("hi there!")
# history.add_user_message("what is your name?")
# history.add_ai_message("my name is siti.")

print(
    chain.invoke(
        {"question": "what is my favorite sport?", "chat_history": history.messages}
    )
)

# app = FastAPI(
#     title="LangChain Server",
#     version="1.0",
#     description="A simple API server using LangChain's Runnable interfaces",
# )

# 3. Adding chain route
# add_routes(
#     app,
#     chain,
#     path="/chain",
# )

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="localhost", port=8000)
