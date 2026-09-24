# -*- coding: utf-8 -*-
"""
Module 6: Agentic RAG - Exercises & Solutions
==============================================

This file contains the completed solution implementation for all exercises in exercises.md.
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

print("Setting up Agentic RAG tools...")
data_dir = os.path.dirname(__file__)
data_path = os.path.join(data_dir, '../../data/synthetic_tickets.json')

with open(data_path, 'r', encoding='utf-8') as f:
    tickets = json.load(f)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
documents = []
for ticket in tickets:
    content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""
    
    doc = Document(
        page_content=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )
    documents.append(doc)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="m6_exercise_agent"
)


# Define Tool Functions
def search_similar_tickets(query: str) -> str:
    if not query or not query.strip():
        return "Error: Empty query."
    
    if "critical" in query.lower():
        results = vectorstore.similarity_search(query, k=3, filter={"priority": "Critical"})
    else:
        results = vectorstore.similarity_search(query, k=3)
    
    if not results:
        return "No tickets found."
    
    output = "Found tickets:\n\n"
    for i, doc in enumerate(results, 1):
        output += f"--- Ticket {i} ---\n{doc.page_content}\n\n"
    return output

def get_ticket_by_id(ticket_id: str) -> str:
    if not ticket_id or not ticket_id.strip():
        return "Error: Provide ticket ID."
    ticket_id = ticket_id.upper().strip()
    if not ticket_id.startswith("TICK-"):
        return f"Error: Invalid ID format '{ticket_id}'."
    
    for ticket in tickets:
        if ticket['ticket_id'] == ticket_id:
            return f"[{ticket['ticket_id']}] {ticket['title']} - {ticket['description']}\nResolution: {ticket['resolution']}"
    return f"Ticket {ticket_id} not found."

def search_by_category(category: str) -> str:
    category = category.strip()
    matching = [t for t in tickets if t['category'].lower() == category.lower()]
    if not matching:
        return f"No tickets in category '{category}'."
    return "\n".join([f"• [{t['ticket_id']}] {t['title']}" for t in matching])

def get_ticket_statistics(input: str = "") -> str:
    total = len(tickets)
    categories = {}
    for t in tickets:
        categories[t['category']] = categories.get(t['category'], 0) + 1
    return f"Total Tickets: {total}\nCategories: {categories}"


# Build Tools Array with enhanced descriptions
tools = [
    Tool(
        name="SearchSimilarTickets",
        func=search_similar_tickets,
        description="Search similar tickets using semantic similarity. Use for troubleshooting 'how to fix' questions."
    ),
    Tool(
        name="GetTicketByID",
        func=get_ticket_by_id,
        description="Retrieve details of a specific ticket by exact ID (e.g. TICK-001)."
    ),
    Tool(
        name="SearchByCategory",
        func=search_by_category,
        description="Find all tickets in a category (Authentication, Database, Payment, etc.)."
    ),
    Tool(
        name="GetTicketStatistics",
        func=get_ticket_statistics,
        description="Get overview statistics of ticket dataset."
    )
]

llm = ChatOpenAI(model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'), temperature=0)
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"]
            }
        }
    }
    for tool in tools
]
llm_with_tools = llm.bind(tools=tool_definitions)


# ============================================================================
# Exercises 1-4 & 6: ReAct Agent Loop with Custom Prompt
# Problem: Build an autonomous agent loop that calls tools dynamically.
# Solution: Loop through model tool calls, execute matching tool functions, feed output back via `ToolMessage`.
# ============================================================================
print("=" * 80)
print("MODULE 6: ReAct Agent Execution Loop")
print("=" * 80)

def run_agent(query: str, max_iterations: int = 5):
    messages = [
        SystemMessage(content="You are an expert support desk assistant. Use tools when needed."),
        HumanMessage(content=query)
    ]
    tools_used = []
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content, tools_used
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"].get("input", "")
            print(f"  🔧 Agent invoking Tool: {tool_name} (Input: '{tool_input}')")
            tools_used.append(tool_name)
            
            tool_output = None
            for tool in tools:
                if tool.name == tool_name:
                    tool_output = tool.func(tool_input)
                    break
            
            messages.append(ToolMessage(content=tool_output or "Tool error", tool_call_id=tool_call["id"]))

    return "Max iterations reached.", tools_used


# Test Single-step and Multi-step queries
q1 = "How do I fix authentication problems?"
ans1, t1 = run_agent(q1)
print(f"\nQuery: '{q1}'\nTools Used: {t1}\nResponse: {ans1[:120]}...\n")

q2 = "Show me ticket TICK-005"
ans2, t2 = run_agent(q2)
print(f"\nQuery: '{q2}'\nTools Used: {t2}\nResponse: {ans2[:120]}...\n")

q3 = "How many payment tickets do we have and what was the resolution for the most recent one?"
ans3, t3 = run_agent(q3)
print(f"\nMulti-step Query: '{q3}'\nTools Used (in sequence): {t3}\nResponse: {ans3[:150]}...\n")

print("=" * 80)
print("MODULE 6 EXERCISES COMPLETE!")
print("=" * 80)
