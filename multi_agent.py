from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
import os
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores import Chroma
# from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

memory = ConversationBufferMemory(memory_key="chat_history")

assistant_llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, api_key=OPENAI_API_KEY)
data_collector_llm = ChatOpenAI(model="gpt-3.5_turbo", temperature=0, api_key=OPENAI_API_KEY)
members = ["Data_collector", "Preferer"]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, chat_history=memory)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def get_supervisor_chain():
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers: {members}. Given the following user request,"
        " respond with the worker to act next based on the user's intent:"
        "\n\nIf the user is asking about car preferences or seeking recommendations or greeting, choose 'Preferer'."
        "\n\nIf the user has decided to purchase a car and needs to provide personal details, choose 'Data_collector'."
        "\n\nEach worker will perform its task and respond. When the task is completed, respond with FINISH."
        "\n\nYou must choose the appropriate worker to complete the task."
    )
    # Our team supervisor is an LLM node. It just picks the next agent to process
    # and decides when the work is completed
    options = ["FINISH"] + members
    # Using openai function calling can make output parsing easier for us
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
        prompt
        | assistant_llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    return supervisor_chain


def get_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    loader = PyPDFLoader("/home/lnv-241/Downloads/Markdown to PDF.pdf")
    pages = loader.load()

    #split text into chunks
    text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks = text_splitter.split_documents(pages)


    db = Chroma.from_documents(text_chunks, embeddings)
    return db


def get_user_info(tool_input):
    return tool_input


def process_user_query(input):
    db = get_db()
    retrive_tool = create_retriever_tool(
        db.as_retriever(k=2),
        name="retrieve_information",
        description="""Useful when welcoming a user or discussing their car preferences, provide tailored top two suggestions from the document. Highlight the recommended options and explain why they align with the user's unique preferences. Do not use this when user is decided to purchase a final car""")


    
    custom_tool = [
        Tool(
            name="get_user_info",
            func=get_user_info,
            description="""Useful when the user finalize the car model to purchase ask user his full name ,email id , and retrive the car selected by the user from chat history and pass them to a function for processing. Once the details are collected conclude the conversation by informing the user about that will send follow-up email and stop conversation""",
            )
    ]


    data_collector_agent = create_agent(data_collector_llm, custom_tool,"Useful when the user decides to purchase a car and given a car preference. Ask user his full name ,email id , and retrive the car selected by the user from chat history and pass them to a function for processing. If all the details provided  then conclude the conversation by informing the user about a follow-up email")
    data_collector_node = functools.partial(agent_node, agent=data_collector_agent, name="Data_collector")

    assistant_agent = create_agent(assistant_llm, [retrive_tool], "Guide users through the process of choosing a car, suggesting suitable models based on their preferences.")
    assistant_node = functools.partial(agent_node, agent=assistant_agent, name="Preferer")

    supervisor_chain = get_supervisor_chain()
    workflow = StateGraph(AgentState)
    workflow.add_node("Data_collector", data_collector_node)
    workflow.add_node("Preferer", assistant_node)
    workflow.add_node("supervisor", supervisor_chain)


    for member in members:
        workflow.add_edge(member, "supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    # Finally, add entrypoint
    workflow.set_entry_point("supervisor")

    graph = workflow.compile()


    for s in graph.stream(
        {
            "messages": [
                HumanMessage(content=f"{input}")
            ]
        }
    ):
        if "__end__" not in s:
            print(s)
            try:
                if "Preferer" in s:
                    res = s['Preferer']['messages'][0].content
                    return res
                if "Data_collector" in s:
                    res = s['Data_collector']['messages'][0].content
                    return res
            except:
                pass
            
            
import streamlit as st
def main():
    st.title("Langchain-Based Streamlit App")

    user_input = st.text_input("Ask something:")
    if user_input:
        response = process_user_query(user_input)
        st.write("Response:")
        st.write(response)

if __name__ == "__main__":
    main()
