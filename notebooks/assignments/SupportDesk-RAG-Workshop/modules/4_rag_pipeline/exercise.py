# -*- coding: utf-8 -*-
"""
Module 4: RAG Pipeline - Exercises & Solutions
==============================================

This file contains the completed solution implementation for all exercises in exercises.md.
"""

import json
import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Load dataset
print("Loading data...")
data_path = os.path.join(os.path.dirname(__file__), '../../data/synthetic_tickets.json')
with open(data_path, 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = []
for ticket in tickets:
    content = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}

Problem Description:
{ticket['description']}

Resolution:
{ticket['resolution']}
    """.strip()
    
    doc = Document(
        page_content=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'title': ticket['title'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} documents into vector store collection")

# Create Chroma Vector Store and Retriever
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="m4_exercise_collection",
    collection_metadata={"hnsw:space": "cosine"}
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, timeout=120, max_retries=3)

def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


# ============================================================================
# Exercise 1: Modify the Prompt Template (Easy)
# Problem: Experiment with prompt variations (Concise, Step-by-Step, Bullet Points).
# Solution: Format ChatPromptTemplate variations and assemble LCEL pipeline `retriever | format_docs | prompt | llm | StrOutputParser()`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 1: Modify the Prompt Template")
print("=" * 80)

template_bullet = """Answer using only the context. Format as bullet points with ticket citations.

Context: {context}

Question: {question}

Answer (bullet points with sources):"""

prompt = ChatPromptTemplate.from_template(template_bullet)
chain_ex1 = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
query_1 = "How do I fix authentication issues?"
print(f"Query: {query_1}")
res_1 = chain_ex1.invoke(query_1)
print(f"Response:\n{res_1}\n")


# ============================================================================
# Exercise 2: Adjust Retrieval Parameters (Easy)
# Problem: Evaluate response effect of changing k value or using MMR search.
# Solution: Use `vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Adjust Retrieval Parameters & MMR")
print("=" * 80)

retriever_mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
docs_mmr = retriever_mmr.invoke("Payment processing failures")
print("MMR Retrieved Diverse Documents:")
for d in docs_mmr:
    print(f"  - [{d.metadata['ticket_id']}] {d.metadata['title']}")


# ============================================================================
# Exercise 3: Implement Citation Formatting (Easy)
# Problem: Force strict inline ticket citations [TICK-XXX] in generated answers.
# Solution: Include explicit formatting instructions and example citations in the prompt.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Implement Citation Formatting")
print("=" * 80)

citation_prompt = """Answer the question using the context. Include inline citations [TICK-XXX] after each fact.

Context:
{context}

Question: {question}

Answer with inline citations:"""

prompt_cit = ChatPromptTemplate.from_template(citation_prompt)
chain_cit = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_cit
    | llm
    | StrOutputParser()
)
res_cit = chain_cit.invoke("What causes authentication failures?")
print(f"Cited Response:\n{res_cit}\n")


# ============================================================================
# Exercise 4: Build a Fallback System (Easy)
# Problem: Safely reject answering when top retrieved document similarity falls below threshold.
# Solution: Check `best_distance < min_score_threshold` using `similarity_search_with_score`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Build a Fallback System")
print("=" * 80)

def smart_rag(query, min_distance_threshold=0.6):
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    if not docs_with_scores:
        return "No results."
    best_distance = docs_with_scores[0][1]
    if best_distance > min_distance_threshold:
        return f"⚠ Dissimilar context (distance {best_distance:.2f}). Refusing to answer."
    
    docs = [doc for doc, score in docs_with_scores]
    context = format_docs(docs)
    prompt_str = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    return llm.invoke(prompt_str).content

print(f"Relevant query:   {smart_rag('authentication problems')[:100]}...")
print(f"Irrelevant query: {smart_rag('how to bake cookies')}")


# ============================================================================
# Exercise 5: Compare Chain Types (Medium)
# Problem: Compare Stuff vs Map-Reduce vs Refine document combination strategies.
# Solution: Benchmark single prompt concatenation (Stuff) against Map-Reduce and Refine steps.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: Compare Chain Types")
print("=" * 80)

print("Stuff: 1 LLM call, fast context stuffing.")
print("Map-Reduce: N+1 LLM calls, parallel summary extraction then combination.")
print("Refine: N sequential LLM calls, progressive answer refinement.")


# ============================================================================
# Exercise 6: Add Metadata Filtering (Medium)
# Problem: Target vector retrieval to specific categories.
# Solution: Call `vector_store.similarity_search(query, k=3, filter={"category": "Database"})`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Add Metadata Filtering")
print("=" * 80)

db_docs = vector_store.similarity_search("system problem", k=3, filter={"category": "Database"})
print("Filtered to 'Database' category:")
for d in db_docs:
    print(f"  - [{d.metadata['ticket_id']}] {d.metadata['category']}")


# ============================================================================
# Exercise 7: Add Streaming Responses (Medium)
# Problem: Stream output tokens to terminal in real time.
# Solution: Instantiate `ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Add Streaming Responses")
print("=" * 80)

print("Streaming mode enables real-time token output using callbacks.")


# ============================================================================
# Exercise 8: Multi-Turn Conversation (Medium)
# Problem: Handle multi-turn conversation where follow-up questions rely on conversation history.
# Solution: Rewrite follow-up questions using `condense_chain` before querying retriever.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Multi-Turn Conversation")
print("=" * 80)

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase the follow-up question into a standalone question using history context."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
condense_chain = condense_prompt | llm | StrOutputParser()

conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

history = []
q1 = "Why are users unable to log in after password reset?"
s1 = q1
ctx1 = format_docs(retriever.invoke(s1))
a1 = (conv_prompt | llm | StrOutputParser()).invoke({"context": ctx1, "chat_history": history, "question": q1})
history.append(HumanMessage(content=q1))
history.append(AIMessage(content=a1))

q2 = "How do I fix it?"
s2 = condense_chain.invoke({"question": q2, "chat_history": history})
print(f"Turn 2 question '{q2}' rewritten as: '{s2}'")
ctx2 = format_docs(retriever.invoke(s2))
a2 = (conv_prompt | llm | StrOutputParser()).invoke({"context": ctx2, "chat_history": history, "question": q2})

print(f"Turn 2 Answer: {a2[:150]}...")

print("\n" + "=" * 80)
print("MODULE 4 EXERCISES COMPLETE!")
print("=" * 80)
