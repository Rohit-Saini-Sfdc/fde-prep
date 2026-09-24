# -*- coding: utf-8 -*-
"""
Module 1: Embeddings & Similarity Search - Exercises & Solutions
=================================================================

This file contains the completed solution implementation for all exercises in exercises.md.
"""

import json
import time
import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

# Load data (used across exercises)
print("Loading data...")
data_path = os.path.join(os.path.dirname(__file__), '../../data/synthetic_tickets.json')
with open(data_path, 'r', encoding='utf-8') as f:
    tickets = json.load(f)

texts = [f"{t['title']}. {t['description']}" for t in tickets]
response = client.embeddings.create(input=texts, model=model)
embeddings = np.array([data.embedding for data in response.data])
print(f"✓ Loaded {len(tickets)} tickets and generated embeddings\n")


# ============================================================================
# Exercise 1: Change the Search Query (Easy)
# Problem: Perform semantic search using a specific issue query.
# Solution: Embed the query and compute cosine similarity against ticket embeddings.
# ============================================================================
print("=" * 80)
print("EXERCISE 1: Change the Search Query")
print("=" * 80)

query = "Database is running very slowly"

query_response = client.embeddings.create(input=[query], model=model)
query_embedding = np.array([query_response.data[0].embedding])
similarities = cosine_similarity(query_embedding, embeddings)[0]

top_indices = np.argsort(similarities)[::-1][:5]

print(f"\nQuery: '{query}'")
print("-" * 50)
for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]
    print(f"#{rank} [{score:.4f}] {ticket['title']}")
    print(f"    Category: {ticket['category']}")


# ============================================================================
# Exercise 2: Adjust the Number of Results (Easy)
# Problem: Inspect top-K results with a larger cutoff (top_k = 10).
# Solution: Slice the sorted similarity indices array to top_k = 10.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Adjust the Number of Results")
print("=" * 80)

top_k = 10

query = "Users can't login after changing password"
query_response = client.embeddings.create(input=[query], model=model)
query_embedding = np.array([query_response.data[0].embedding])
similarities = cosine_similarity(query_embedding, embeddings)[0]

top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"\nQuery: '{query}'")
print(f"Showing top {top_k} results:")
print("-" * 50)

below_threshold_rank = None
for rank, idx in enumerate(top_indices, 1):
    score = similarities[idx]
    ticket = tickets[idx]
    print(f"#{rank} [{score:.4f}] {ticket['title']}")
    
    if below_threshold_rank is None and score < 0.5:
        below_threshold_rank = rank

print(f"\n→ Score drops below 0.5 at rank: {below_threshold_rank or 'Never (all above 0.5)'}")


# ============================================================================
# Exercise 3: Add a Similarity Threshold (Easy)
# Problem: Filter out low-relevance search results that fall below a minimum score.
# Solution: Add a condition (`if score < threshold: continue`) inside the retrieval loop.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Add a Similarity Threshold")
print("=" * 80)

def search_with_threshold(query_text, threshold=0.5):
    query_response = client.embeddings.create(input=[query_text], model=model)
    query_embedding = np.array([query_response.data[0].embedding])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    top_indices = np.argsort(similarities)[::-1][:10]
    
    print(f"\nQuery: '{query_text}' (threshold: {threshold})")
    print("-" * 50)
    
    count = 0
    for rank, idx in enumerate(top_indices, 1):
        ticket = tickets[idx]
        score = similarities[idx]
        
        # Skip results below threshold
        if score < threshold:
            continue
        
        count += 1
        print(f"#{rank} [{score:.4f}] {ticket['title']}")
    
    if count == 0:
        print("No relevant tickets found above threshold.")
    else:
        print(f"\n→ {count} tickets above threshold")

search_with_threshold("Users can't login after changing password")
search_with_threshold("How to make pizza")


# ============================================================================
# Exercise 4: Compare Two Queries (Easy)
# Problem: Observe how different semantic queries rank tickets differently.
# Solution: Loop through multiple query strings and extract top-1 matches for each.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Compare Two Queries")
print("=" * 80)

query1 = "Login authentication failed"
query2 = "Slow database performance"

for q in [query1, query2]:
    resp = client.embeddings.create(input=[q], model=model)
    q_emb = np.array([resp.data[0].embedding])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argmax(sims)
    
    print(f"\nQuery: '{q}'")
    print(f"  Best match: {tickets[top_idx]['title']}")
    print(f"  Category: {tickets[top_idx]['category']}")
    print(f"  Score: {sims[top_idx]:.4f}")


# ============================================================================
# Exercise 5: Test Semantic Understanding (Medium)
# Problem: Prove that embeddings capture conceptual meaning rather than keyword overlap.
# Solution: Compute pairwise cosine similarity matrix across synonymous vs distinct texts.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: Test Semantic Understanding")
print("=" * 80)

test_texts = [
    "User authentication failed",      # Original
    "Login credentials rejected",       # Same meaning, different words
    "Cannot sign in to account",        # Same meaning, different words
    "Database connection timeout",      # Different topic
]

resp = client.embeddings.create(input=test_texts, model=model)
test_embeddings = np.array([data.embedding for data in resp.data])
similarity_matrix = cosine_similarity(test_embeddings)

print("\nSimilarity Matrix Analysis:")
print("-" * 60)
for i, text1 in enumerate(test_texts):
    for j, text2 in enumerate(test_texts):
        if i < j:
            sim = similarity_matrix[i][j]
            relation = "SIMILAR" if sim > 0.7 else "DIFFERENT"
            print(f"'{text1}' vs '{text2}'\n  -> Score: {sim:.3f} [{relation}]")

print("\n→ Auth/Login texts have HIGH similarity despite different words.")
print("→ Database text has LOW similarity (different domain).")


# ============================================================================
# Exercise 6: Filter by Category (Medium)
# Problem: Narrow down semantic search results to a specific metadata category.
# Solution: Apply `if category_filter and ticket['category'] != category_filter: continue`.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Filter by Category")
print("=" * 80)

def search_with_category(query, category_filter=None, top_k=5):
    resp = client.embeddings.create(input=[query], model=model)
    query_emb = np.array([resp.data[0].embedding])
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    results = []
    for idx in np.argsort(similarities)[::-1]:
        ticket = tickets[idx]
        if category_filter and ticket['category'] != category_filter:
            continue
        results.append((ticket, similarities[idx]))
        if len(results) >= top_k:
            break
    return results

print("\nAll categories for 'login problem':")
for ticket, score in search_with_category("login problem"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")

print("\nOnly 'Authentication' category for 'login problem':")
for ticket, score in search_with_category("login problem", category_filter="Authentication"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")


# ============================================================================
# Exercise 7: Batch vs Single Embedding (Medium)
# Problem: Measure efficiency difference between single API calls vs batch requests.
# Solution: Use `time.time()` around sequential single calls vs one batch call.
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Batch vs Single Embedding")
print("=" * 80)

batch_texts = [
    "Password reset not working",
    "Database connection timeout", 
    "App crashes on startup",
    "Payment declined error",
    "Email notifications delayed",
]

start = time.time()
for text in batch_texts:
    _ = client.embeddings.create(input=[text], model=model)
time_slow = time.time() - start
print(f"Single API calls time: {time_slow:.2f} seconds")

start = time.time()
_ = client.embeddings.create(input=batch_texts, model=model)
time_fast = time.time() - start
print(f"Batch API call time:   {time_fast:.2f} seconds")

speedup = time_slow / time_fast if time_fast > 0 else float('inf')
print(f"\n✓ Batch API is {speedup:.1f}x faster!")


# ============================================================================
# Bonus Exercise: Similarity Matrix Heatmap (Challenge)
# Problem: Visualize pairwise document similarity as a heatmap graphic.
# Solution: Use `plt.imshow` with `RdYlGn` colormap and label ticket similarity scores.
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Similarity Matrix Heatmap")
print("=" * 80)

sample_tickets = tickets[:10]
sample_texts = [t['title'] for t in sample_tickets]
resp = client.embeddings.create(input=sample_texts, model=model)
sample_embeddings = np.array([data.embedding for data in resp.data])
sim_matrix = cosine_similarity(sample_embeddings)

plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='Cosine Similarity')
labels = [f"{t['ticket_id']}" for t in sample_tickets]
plt.xticks(range(10), labels, rotation=45, ha='right')
plt.yticks(range(10), labels)
plt.title('Ticket Similarity Matrix (First 10 Tickets)')
plt.tight_layout()
output_img = os.path.join(os.path.dirname(__file__), 'exercise_heatmap.png')
plt.savefig(output_img, dpi=150)
print(f"✓ Saved similarity heatmap to {output_img}")

print("\n" + "=" * 80)
print("MODULE 1 EXERCISES COMPLETE!")
print("=" * 80)
