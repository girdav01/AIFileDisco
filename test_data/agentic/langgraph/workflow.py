from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

def agent_node(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages': [response]}

workflow = StateGraph(AgentState)
workflow.add_node('agent', agent_node)
workflow.add_node('tool_node', tool_node)
workflow.set_entry_point('agent')
