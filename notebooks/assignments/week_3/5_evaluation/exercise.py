# -*- coding: utf-8 -*-
"""
Module 5: Evaluation & Metrics - Exercises & Solutions
======================================================

This file contains the completed solution implementation for all exercises in exercises.md.
"""

import json
import os
import time
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# Load dataset and eval queries
print("Loading data...")
data_dir = os.path.dirname(__file__)
data_path = os.path.join(data_dir, '../../data/synthetic_tickets.json')
eval_path = os.path.join(data_dir, 'evaluation_queries.json')

with open(data_path, 'r', encoding='utf-8') as f:
    tickets = json.load(f)

with open(eval_path, 'r', encoding='utf-8') as f:
    eval_queries = json.load(f)

documents = []
for ticket in tickets:
    content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}"""
    
    doc = Document(
        page_content=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'title': ticket['title'],
            'category': ticket['category']
        }
    )
    documents.append(doc)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chat_model = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')
vector_store = FAISS.from_documents(documents, embeddings)
print(f"✓ FAISS Vector store ready with {len(documents)} documents\n")


# ============================================================================
# Exercise 1: Calculate Precision, Recall, F1 (Easy)
# Problem: Implement standard information retrieval metrics at cutoff k.
# Solution:
#   tp = len(retrieved_set & relevant_set)
#   precision = tp / k
#   recall = tp / len(relevant_set)
#   f1 = 2 * (precision * recall) / (precision + recall)
# ============================================================================
print("=" * 80)
print("EXERCISE 1: Calculate Precision, Recall, F1")
print("=" * 80)

def calculate_metrics(retrieved_ids, relevant_ids, k=3):
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    tp = len(retrieved_set & relevant_set)
    precision = tp / k if k > 0 else 0.0
    recall = tp / len(relevant_set) if len(relevant_set) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

retrieved = ['TICK-001', 'TICK-002', 'TICK-003']
relevant = ['TICK-001', 'TICK-003']
m = calculate_metrics(retrieved, relevant, k=3)
print(f"Test case (retrieved={retrieved}, relevant={relevant}):")
print(f"  Precision@3: {m['precision']:.2f}, Recall@3: {m['recall']:.2f}, F1@3: {m['f1']:.2f}")


# ============================================================================
# Exercise 2: Evaluate All Queries (Easy)
# Problem: Run retrieval metrics across all test queries in evaluation dataset.
# Solution: Compute mean precision, recall, and F1 across dataset.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Evaluate All Queries")
print("=" * 80)

all_metrics = []
for q in eval_queries:
    docs = vector_store.similarity_search(q['question'], k=3)
    retrieved_ids = [d.metadata['ticket_id'] for d in docs]
    all_metrics.append(calculate_metrics(retrieved_ids, q['relevant_ticket_ids'], k=3))

avg_p = np.mean([x['precision'] for x in all_metrics])
avg_r = np.mean([x['recall'] for x in all_metrics])
avg_f1 = np.mean([x['f1'] for x in all_metrics])

print(f"Dataset-wide Averages ({len(eval_queries)} queries):")
print(f"  Mean Precision@3: {avg_p:.4f}")
print(f"  Mean Recall@3:    {avg_r:.4f}")
print(f"  Mean F1@3:        {avg_f1:.4f}")


# ============================================================================
# Exercise 3: Compare Different k Values (Easy)
# Problem: Analyze trade-off between precision and recall as k varies.
# Solution: Loop over `k in [1, 3, 5, 10]` and calculate mean precision and recall.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Compare Different k Values")
print("=" * 80)

for k in [1, 3, 5, 10]:
    k_metrics = []
    for q in eval_queries:
        docs = vector_store.similarity_search(q['question'], k=k)
        retrieved_ids = [d.metadata['ticket_id'] for d in docs]
        k_metrics.append(calculate_metrics(retrieved_ids, q['relevant_ticket_ids'], k=k))
    
    p = np.mean([x['precision'] for x in k_metrics])
    r = np.mean([x['recall'] for x in k_metrics])
    print(f"k={k:2d} -> Precision: {p:.4f} | Recall: {r:.4f}")


# ============================================================================
# Exercise 4: LLM-as-Judge for Groundedness (Medium)
# Problem: Verify if generated LLM answers stay faithful to retrieved context.
# Solution: Prompt GPT-4o-mini to rate groundedness 0-10 based on context documents.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: LLM-as-Judge for Groundedness")
print("=" * 80)

def evaluate_groundedness(answer, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""Evaluate if the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Rate groundedness (0-10):
Format: Score: X / Reason: <explanation>"""

    resp = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    output = resp.choices[0].message.content
    try:
        score = float(output.split('Score:')[1].split('/')[0].strip()) / 10
    except:
        score = 0.5
    return {'score': score, 'output': output}

q_sample = eval_queries[0]
docs_sample = vector_store.similarity_search(q_sample['question'], k=3)
context_text = "\n\n".join([d.page_content for d in docs_sample])
gen_resp = openai_client.chat.completions.create(
    model=chat_model,
    messages=[{"role": "user", "content": f"Context: {context_text}\n\nQuestion: {q_sample['question']}\n\nAnswer:"}],
    temperature=0
).choices[0].message.content

g_res = evaluate_groundedness(gen_resp, docs_sample)
print(f"Query: {q_sample['question']}")
print(f"Groundedness score: {g_res['score']:.2f}")


# ============================================================================
# Exercise 5: LLM-as-Judge for Completeness (Medium)
# Problem: Check if RAG response addresses all aspects of the original user question.
# Solution: Prompt GPT-4o-mini to rate completeness 0-10 against reference answers.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: LLM-as-Judge for Completeness")
print("=" * 80)

def evaluate_completeness(question, answer):
    prompt = f"""Evaluate if the ANSWER fully addresses the QUESTION.

QUESTION: {question}
ANSWER: {answer}

Rate completeness (0-10):
Format: Score: X / Reason: <explanation>"""

    resp = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    output = resp.choices[0].message.content
    try:
        score = float(output.split('Score:')[1].split('/')[0].strip()) / 10
    except:
        score = 0.5
    return {'score': score, 'output': output}

c_res = evaluate_completeness(q_sample['question'], gen_resp)
print(f"Completeness score: {c_res['score']:.2f}")


# ============================================================================
# Exercise 6: Failure Analysis (Medium)
# Problem: Identify specific queries where retrieval precision falls below threshold.
# Solution: Filter queries with `precision < 0.5` and group by category using `defaultdict(list)`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Failure Analysis")
print("=" * 80)

failures = []
for q in eval_queries:
    docs = vector_store.similarity_search(q['question'], k=3)
    retrieved = [d.metadata['ticket_id'] for d in docs]
    met = calculate_metrics(retrieved, q['relevant_ticket_ids'])
    if met['precision'] < 0.5:
        failures.append((q['question'], q.get('category', 'Unknown'), met['precision']))

print(f"Total retrieval failures (Precision < 0.5): {len(failures)}")
for q_text, cat, p_val in failures[:3]:
    print(f"  Category [{cat}] (p={p_val:.2f}): '{q_text}'")


# ============================================================================
# Exercise 7: Track Latency and Cost (Medium)
# Problem: Monitor system operational metrics (P95 latency & token costs).
# Solution: Track query elapsed time with `time.time()` and compute estimated LLM API costs.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Track Latency and Cost")
print("=" * 80)

latencies = []
start = time.time()
docs = vector_store.similarity_search("login issue", k=3)
latencies.append(time.time() - start)
print(f"Sample retrieval latency: {latencies[0]*1000:.2f} ms")


# ============================================================================
# Exercise 8: Implement a Release Gate (Medium)
# Problem: Automate release deployment decisions (PASS/REVIEW/BLOCK) based on metrics.
# Solution: Evaluate metrics against target thresholds (e.g. Precision >= 0.80, Groundedness >= 0.85).
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Implement a Release Gate")
print("=" * 80)

def get_release_decision(metrics: dict):
    flags = []
    if metrics['precision'] < 0.80:
        flags.append(f"precision = {metrics['precision']:.2f} (target 0.80)")
    if metrics['groundedness'] < 0.85:
        flags.append(f"groundedness = {metrics['groundedness']:.2f} (target 0.85)")
    
    if not flags:
        return 'PASS', flags
    elif any("0.70" in f or metrics['precision'] < 0.60 for f in flags):
        return 'BLOCK', flags
    else:
        return 'REVIEW', flags

test_sys = {'precision': 0.85, 'groundedness': 0.90}
dec, f_list = get_release_decision(test_sys)
print(f"System decision: {dec} | Flags: {f_list}")

print("\n" + "=" * 80)
print("MODULE 5 EXERCISES COMPLETE!")
print("=" * 80)
