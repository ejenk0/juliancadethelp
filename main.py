from langchain import OpenAI, LLMMathChain
from langchain.agents import (
    initialize_agent,
    Tool,
    AgentExecutor,
    create_pandas_dataframe_agent,
)
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
import os
import chainlit as cl
import pandas as pd

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


@cl.on_chat_start
def start():
    llm = ChatOpenAI(
        temperature=0, streaming=True, model="gpt-4"
    )  # this one is just for speaking
    llm1 = OpenAI(temperature=0, streaming=True)  # this one is for math
    llm2 = OpenAI(temperature=0, streaming=True)  # pandas
    llm_math_chain = LLMMathChain.from_llm(llm=llm1, verbose=True)
    df = pd.read_csv("temp.csv")
    # could point the file to where the table is stored in memory, however i still need to declare the location AFTER the agent is initiated
    pandas = create_pandas_dataframe_agent(llm2, df, verbose=True)
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    tools = [
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="useful for gathering information about anything. data retrieved will also need to be trimmed",
        ),
        Tool(
            name="Pandas",
            func=pandas.run,
            description="useful for analysing data and creating graphs. input to this tool must be a single string.",
        ),
    ]
    agent = initialize_agent(
        tools, llm, agent="chat-zero-shot-react-description", verbose=True
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])
