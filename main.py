from langchain import OpenAI, LLMMathChain
from langchain.agents import (
    initialize_agent,
    Tool,
    AgentExecutor,
    create_pandas_dataframe_agent,
    create_csv_agent,
)
from langchain.chat_models import ChatOpenAI
from langchain.tools import (
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
    StructuredTool,
)
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
import os
import chainlit as cl
import pandas as pd
import wikipedia as wk

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

WIKIPEDIA_MAX_QUERY_LENGTH = 300


def get_wikipedia_tables(query: str, result_index: int = 0) -> str:
    """Search Wikipedia for a page matching the query and return a list of
    all tables in the page as a string or an error
    As only 4000 characters can be returned at a time, you can use result_index
    to get the next 4000 in your next query.
    For example, result_index=0 returns up to the first 4000 characters.
    resul_index=1 returns characters 4001-8000 etc."""
    try:
        page_title = wk.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH])[0]
    except IndexError:
        return "No valid pages found for query"
    page = wk.page(page_title)
    dataframes = pd.read_html(page.html())
    result = str(dataframes)
    result_len = len(result)

    if result_len < (result_index * 4000):
        return "Result index out of range. Total result length:" + str(
            result_len
        )
    else:
        result = result[result_index * 4000 :]

    result_len = len(result)

    if result_len > 4000:
        result = result[:4000]

    # DEBUG
    print("return result of length", len(result))

    return result


def write_to_file(file_name: str, file_content: str) -> bool:
    """Writes file_contents to a new file with the given file name.
    Returns a boolean indicating success (True) or failure (False)"""
    try:
        with open(
            os.path.join(os.getcwd(), "ai_written_files", file_name), "w"
        ) as f:
            f.write(file_content)
    except:
        return False
    return True


def query_data_set(csv_file_name: str, pandas_agent_query: str) -> str:
    """Creates an AI pandas agent to read the given csv file.
    The query shuold be a plain english description of the data you wish to
    retrieve.
    The result is returned or an error message if something went wrong.
    """
    try:
        pandas_agent = create_csv_agent(
            OpenAI(temperature=0),
            os.path.join(os.getcwd(), "ai_written_files", csv_file_name),
            verbose=True,
        )
    except:
        return "ERROR: Something went wrong when creating the CSV agent. Check the CSV file exists!"

    return pandas_agent.run(pandas_agent_query)


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
            description="useful for when you need to answer questions about current events. You should ask targeted questions. Accepts a single string as input",
        ),
        Tool(
            name="WikipediaSummary",
            func=wikipedia.run,
            description="useful for gathering information about anything. data retrieved will also need to be trimmed",
        ),
        # Tool(
        #     name="Pandas",
        #     func=pandas.run,
        #     description="useful for analysing data and creating graphs. input to this tool must be a single, english language string to be passed to an AI agent which can query the dataset and return the result.",
        # ),
        StructuredTool.from_function(write_to_file),
        StructuredTool.from_function(query_data_set),
        StructuredTool.from_function(get_wikipedia_tables),
    ]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    result = await cl.make_async(agent.run)(message, callbacks=[cb])
    await cl.Message(result).send()
