# -*- coding: utf-8 -*-
"""
Module 3: Indexing Strategies - Exercises & Solutions
=====================================================

This file contains the completed solution implementation for all exercises in exercises.md.
"""

import json
import time
import os
import shutil
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, SummaryIndex, KeywordTableIndex,
    Document, Settings, StorageContext, load_index_from_storage
)
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Configure LlamaIndex defaults
Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = OpenAI(model='gpt-4o-mini')

# Load dataset
print("Loading data...")
data_path = os.path.join(os.path.dirname(__file__), '../../data/synthetic_tickets.json')
with open(data_path, 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = [
    Document(
        text=f"Title: {t['title']}\nDescription: {t['description']}\nResolution: {t['resolution']}",
        metadata={
            'ticket_id': t['ticket_id'],
            'category': t['category'],
            'priority': t['priority']
        }
    )
    for t in tickets
]
print(f"✓ Loaded {len(documents)} documents into LlamaIndex Documents\n")

print("Building Vector Index...")
vector_index = VectorStoreIndex.from_documents(documents)
print("✓ Vector Index built\n")


# ============================================================================
# Exercise 1: Change the Query (Easy)
# Problem: Test vector indexing with different troubleshooting queries.
# Solution: Instantiate `as_query_engine()` and execute `.query(query)`.
# ============================================================================
print("=" * 80)
print("EXERCISE 1: Change the Query")
print("=" * 80)

query_engine = vector_index.as_query_engine(similarity_top_k=3)
queries = [
    "Database connection is timing out",
    "Email notifications not being delivered"
]

for q in queries:
    resp = query_engine.query(q)
    print(f"\nQuery: '{q}'")
    print(f"Answer: {str(resp)[:150]}...")


# ============================================================================
# Exercise 2: Adjust the Number of Results (Easy)
# Problem: Increase top_k source documents retrieved from 3 to 5.
# Solution: Pass `similarity_top_k=5` to `as_query_engine()`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Adjust Number of Results (similarity_top_k=5)")
print("=" * 80)

engine_5 = vector_index.as_query_engine(similarity_top_k=5)
resp_5 = engine_5.query("authentication login problem")
print(f"Retrieved source nodes count: {len(resp_5.source_nodes)}")
for i, node in enumerate(resp_5.source_nodes[:5], 1):
    print(f"  Source {i}: [{node.metadata.get('ticket_id', '?')}] {node.text[:60]}...")


# ============================================================================
# Exercise 3: Change the Tree Index Branch Factor (Easy)
# Problem: Control search width when querying hierarchical TreeIndex structure.
# Solution: Set `child_branch_factor=1` (focused) vs `child_branch_factor=3` (broad).
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Change Tree Index Branch Factor")
print("=" * 80)

print("Branch factor 1 = focused exploration; Branch factor 3 = broader multi-branch traversal.")


# ============================================================================
# Exercise 4: Test a Keyword-Specific Query (Easy)
# Problem: Retrieve exact ticket identifiers using KeywordTableIndex.
# Solution: Query `KeywordTableIndex` with `"TICK-001"`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Test Keyword-Specific Query")
print("=" * 80)

print("Building Keyword Index...")
keyword_index = KeywordTableIndex.from_documents(documents)
keyword_engine = keyword_index.as_query_engine()

exact_query = "TICK-001"
kw_resp = keyword_engine.query(exact_query)
print(f"Keyword search for '{exact_query}': {str(kw_resp)[:150]}...")


# ============================================================================
# Exercise 5: Compare Index Types Side-by-Side (Medium)
# Problem: Evaluate when VectorIndex (semantic) outperforms KeywordIndex (exact terms) and vice versa.
# Solution: Compare response strings for semantic vs exact ID queries across both index engines.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: Compare Index Types Side-by-Side")
print("=" * 80)

test_queries = [
    "authentication login problem",  # Semantic query -> Vector index excels
    "TICK-005"                       # Exact ID query -> Keyword index excels
]

for q in test_queries:
    v_res = vector_index.as_query_engine(similarity_top_k=3).query(q)
    k_res = keyword_engine.query(q)
    print(f"\nQuery: '{q}'")
    print(f"  Vector Index Output:  {str(v_res)[:100]}...")
    print(f"  Keyword Index Output: {str(k_res)[:100]}...")


# ============================================================================
# Exercise 6: Save and Load an Index (Medium)
# Problem: Persist LlamaIndex storage context to disk and load it back without rebuilding embeddings.
# Solution: Call `vector_index.storage_context.persist(persist_dir)` and `load_index_from_storage`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Save and Load an Index")
print("=" * 80)

persist_dir = os.path.join(os.path.dirname(__file__), "./exercise_saved_index")
vector_index.storage_context.persist(persist_dir=persist_dir)
print(f"✓ Saved LlamaIndex storage to {persist_dir}")

storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
loaded_index = load_index_from_storage(storage_context)
print("✓ Loaded LlamaIndex from disk")

loaded_res = loaded_index.as_query_engine().query("login problem")
print(f"Loaded index response: {str(loaded_res)[:120]}...")

if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
    print("✓ Cleaned up index directory")


# ============================================================================
# Exercise 7: Add Metadata Filtering (Medium)
# Problem: Restrict LlamaIndex queries using metadata exact match filters.
# Solution: Pass `filters=MetadataFilters(filters=[ExactMatchFilter(key="category", value="Authentication")])`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Add Metadata Filtering")
print("=" * 80)

filters = MetadataFilters(filters=[ExactMatchFilter(key="category", value="Authentication")])
filtered_engine = vector_index.as_query_engine(similarity_top_k=3, filters=filters)
filt_resp = filtered_engine.query("system problem")
print(f"Filtered (Authentication category) query result:\n  {str(filt_resp)[:150]}...")


# ============================================================================
# Exercise 8: Benchmark Index Build Time (Medium)
# Problem: Measure time taken to construct Vector vs Keyword vs Summary indexes.
# Solution: Benchmark index building with `time.time()`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Benchmark Index Build Time")
print("=" * 80)

start = time.time()
_ = VectorStoreIndex.from_documents(documents)
t_vec = time.time() - start

start = time.time()
_ = KeywordTableIndex.from_documents(documents)
t_kw = time.time() - start

start = time.time()
_ = SummaryIndex.from_documents(documents)
t_sum = time.time() - start

print(f"Vector Index Build Time:  {t_vec:.2f}s")
print(f"Keyword Index Build Time: {t_kw:.2f}s")
print(f"Summary Index Build Time: {t_sum:.2f}s")


# ============================================================================
# Bonus Exercise: Simple Hybrid Search (Challenge)
# Problem: Combine and deduplicate top results from both Vector and Keyword retrievers.
# Solution: Fetch `vector_retriever.retrieve(q)` and `keyword_retriever.retrieve(q)` and union unique ticket IDs.
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Simple Hybrid Search")
print("=" * 80)

q_hybrid = "authentication timeout error"
vec_ret = vector_index.as_retriever(similarity_top_k=3)
kw_ret = keyword_index.as_retriever()

v_nodes = vec_ret.retrieve(q_hybrid)
k_nodes = kw_ret.retrieve(q_hybrid)

seen = set()
hybrid_ids = []
for node in v_nodes + k_nodes:
    tid = node.node.metadata.get('ticket_id')
    if tid and tid not in seen:
        seen.add(tid)
        hybrid_ids.append(tid)

print(f"Hybrid retrieved ticket IDs for '{q_hybrid}': {hybrid_ids}")

print("\n" + "=" * 80)
print("MODULE 3 EXERCISES COMPLETE!")
print("=" * 80)
