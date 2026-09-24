# -*- coding: utf-8 -*-
"""
Module 2: Chunking & Vector Stores - Exercises & Solutions
==========================================================

This file contains the completed solution implementation for all exercises in exercises.md.
"""

import json
import os
import time
import shutil
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Load dataset
print("Loading data...")
data_path = os.path.join(os.path.dirname(__file__), '../../data/synthetic_tickets.json')
with open(data_path, 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = []
for ticket in tickets:
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()
    
    doc = Document(
        page_content=full_text,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} documents\n")


# ============================================================================
# Exercise 1: Change the Chunk Size (Easy)
# Problem: Observe the effect of increasing chunk size on total chunk count.
# Solution: Compare CharacterTextSplitter with chunk_size=200 vs chunk_size=500.
# ============================================================================
print("=" * 80)
print("EXERCISE 1: Change the Chunk Size")
print("=" * 80)

splitter_200 = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
chunks_200 = splitter_200.split_documents(documents)

splitter_500 = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
chunks_500 = splitter_500.split_documents(documents)

print(f"Original (200 chars): {len(chunks_200)} chunks")
print(f"New (500 chars):      {len(chunks_500)} chunks")
print("→ Larger chunk size produces fewer, more context-rich chunks.")


# ============================================================================
# Exercise 2: Change the Search Query (Easy)
# Problem: Query the vector store with distinct domain-specific topics.
# Solution: Perform similarity search on Chroma with new queries.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Change the Search Query")
print("=" * 80)

store = Chroma.from_documents(documents, embeddings, collection_name="ex2_store")
queries = [
    "Database is timing out frequently",
    "Email notifications not working",
    "Payment processing fails"
]

for q in queries:
    results = store.similarity_search(q, k=1)
    if results:
        print(f"\nQuery: '{q}'")
        print(f"  Best Match ID: {results[0].metadata['ticket_id']}")
        print(f"  Category:      {results[0].metadata['category']}")


# ============================================================================
# Exercise 3: Adjust Number of Results (Easy)
# Problem: Retrieve top-5 results instead of top-3 to observe lower rank relevance.
# Solution: Set k=5 in `store.similarity_search(query, k=5)`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Adjust Number of Results (k=5)")
print("=" * 80)

query = "Authentication problems"
results_5 = store.similarity_search(query, k=5)

print(f"Top 5 results for '{query}':")
for i, doc in enumerate(results_5, 1):
    print(f"  #{i}. [{doc.metadata['ticket_id']}] [{doc.metadata['category']}] {doc.page_content[:50]}...")


# ============================================================================
# Exercise 4: Try Different Metadata Filters (Easy)
# Problem: Restrict similarity search to specific ticket categories.
# Solution: Pass `filter={"category": "Database"}` to `similarity_search`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Try Different Metadata Filters")
print("=" * 80)

for category in ["Authentication", "Database", "Performance"]:
    results = store.similarity_search("system not working", k=2, filter={"category": category})
    print(f"\nCategory Filter: '{category}'")
    for doc in results:
        print(f"  [{doc.metadata['ticket_id']}]: {doc.page_content[:50]}...")


# ============================================================================
# Exercise 5: Compare Chunk Sizes (Medium)
# Problem: Benchmark chunk counts and average lengths across chunk size choices.
# Solution: Iterate over `[100, 200, 300, 500, 1000]` with `RecursiveCharacterTextSplitter`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: Compare Chunk Sizes")
print("=" * 80)

print("Chunk Size | # Chunks | Avg Chunk Length")
print("-" * 45)

for size in [100, 200, 300, 500, 1000]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 10)
    chunks = splitter.split_documents(documents)
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
    print(f"{size:>10} | {len(chunks):>8} | {avg_len:>16}")


# ============================================================================
# Exercise 6: Add Similarity Scores to Results (Medium)
# Problem: Inspect numeric distance/similarity scores alongside returned chunks.
# Solution: Use `similarity_search_with_score(query, k=3)`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Add Similarity Scores to Results")
print("=" * 80)

query = "login authentication error"
results_with_scores = store.similarity_search_with_score(query, k=3)

print(f"Results with L2 Distance (lower = more similar):")
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"  #{i} - Distance: {score:.4f} | Ticket: {doc.metadata['ticket_id']}")


# ============================================================================
# Exercise 7: Filter by Multiple Conditions (Medium)
# Problem: Perform multi-condition metadata filtering (e.g. Category AND Priority).
# Solution: Use Chroma's `$and` filter logic: `{"$and": [{"category": "Authentication"}, {"priority": "High"}]}`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Filter by Multiple Conditions")
print("=" * 80)

combined_filter = {"$and": [{"category": "Authentication"}, {"priority": "High"}]}
results_combined = store.similarity_search("system issue", k=3, filter=combined_filter)

print("Results for Category=Authentication AND Priority=High:")
for doc in results_combined:
    print(f"  [{doc.metadata['priority']}] [{doc.metadata['category']}] {doc.metadata['ticket_id']}")


# ============================================================================
# Exercise 8: Save and Load Vector Store (Medium)
# Problem: Persist a vector store to disk and load it without re-computing embeddings.
# Solution: Pass `persist_directory` to Chroma, then instantiate `Chroma(persist_directory=...)`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Save and Load Vector Store")
print("=" * 80)

persist_dir = os.path.join(os.path.dirname(__file__), "./exercise_chroma_db")
saved_store = Chroma.from_documents(documents, embeddings, collection_name="ex8", persist_directory=persist_dir)
print(f"✓ Saved vector store to {persist_dir}")

loaded_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name="ex8")
print("✓ Loaded vector store from disk")

res = loaded_store.similarity_search("login problem", k=2)
print(f"Retrieved from loaded store: {[d.metadata['ticket_id'] for d in res]}")

# Cleanup DB directory
del loaded_store
del saved_store
if os.path.exists(persist_dir):
    try:
        shutil.rmtree(persist_dir)
        print("✓ Cleaned up test directory")
    except Exception:
        pass


# ============================================================================
# Bonus Exercise: Semantic vs Fixed Chunking (Challenge)
# Problem: Compare fixed-size chunking against semantic split points based on topic shifts.
# Solution: Use `SemanticChunker` with `breakpoint_threshold_type="percentile"`.
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Semantic vs Fixed Chunking")
print("=" * 80)

sample_text = """
Authentication Section: User login uses OAuth 2.0 with JWT tokens. Passwords must be hashed using bcrypt.

Database Section: PostgreSQL connection pool handles maximum 50 concurrent connections. Queries timing out after 30s.
""".strip()

doc_sample = Document(page_content=sample_text)
fixed_sp = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=10)
semantic_sp = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

f_chunks = fixed_sp.split_documents([doc_sample])
s_chunks = semantic_sp.split_documents([doc_sample])

print(f"Fixed-size chunk count: {len(f_chunks)}")
print(f"Semantic chunk count:   {len(s_chunks)}")

print("\n" + "=" * 80)
print("MODULE 2 EXERCISES COMPLETE!")
print("=" * 80)
