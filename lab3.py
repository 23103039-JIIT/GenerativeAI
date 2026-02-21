# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np
import string

"""Phase 1: Loading a Pre-Trained Embedding Model"""
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
print(f"Model '{model_name}' loaded successfully!")


"""Phase 2: Generating Word Embeddings
text = input("Enter words separated by commas:\n")
words = [word.strip() for word in text.split(",")]

# Generate embeddings
embeddings = model.encode(words)

# Print embedding vector size
for word, vector in zip(words, embeddings):
    print(f"\nWord: {word}")
    print(f"Embedding vector size: {len(vector)}")

"""


"""Phase 3: Word Semantic Similarity
word1 = input("Enter first word: ").strip()
word2 = input("Enter second word: ").strip()

# Generate embeddings
embedding1 = model.encode(word1)
embedding2 = model.encode(word2)

# Compute cosine similarity
cosine_similarity = np.dot(embedding1, embedding2) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
)
print(f"\nCosine Similarity between '{word1}' and '{word2}': {cosine_similarity:.4f}")
"""


"""Phase 4: Sentence Embedding Generation
sentence = input("Enter a sentence:\n")

# Generate sentence embedding
embedding = model.encode(sentence)

# Display embedding dimension
print("\nSentence Embedding Generated Successfully!")
print("Embedding Dimension:", len(embedding))
"""


"""Phase 5: Sentence Similarity Comparison
sentence1 = input("Enter first sentence:\n")
sentence2 = input("Enter second sentence:\n")

# Generate embeddings
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

# Compute cosine similarity
similarity = np.dot(embedding1, embedding2) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
)
print(f"\nSemantic Similarity Score: {similarity:.4f}")

# Interpret result
threshold = 0.6 
if similarity >= threshold:
    print("Interpretation: The sentences are SEMANTICALLY SIMILAR.")
else:
    print("Interpretation: The sentences are NOT SEMANTICALLY SIMILAR.")
"""


"""Phase 6: Building a Semantic Document Store"""
documents = [
    "Artificial intelligence is transforming healthcare.",
    "Machine learning helps systems learn from data.",
    "Natural language processing enables computers to understand text.",
    "Deep learning is a subset of machine learning."
]

# Generate embeddings for each document
doc_embeddings = model.encode(documents)

# Store document-embedding pairs
semantic_store = []

for doc, embedding in zip(documents, doc_embeddings):
    semantic_store.append({
        "document": doc,
        "embedding": embedding
    })

print("Semantic Document Store Created Successfully!")
print(f"Total Documents Stored: {len(semantic_store)}")
print(f"Embedding Dimension: {len(semantic_store[0]['embedding'])}")


"""Phase 7: Semantic Search Implementation"""
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

def semantic_search(query):
    # Generate query embedding
    query_embedding = model.encode(query)
    similarities = []

    for doc_embedding in doc_embeddings:
        score = cosine_similarity(query_embedding, doc_embedding)
        similarities.append(score)

    # Retrieve most relevant document
    best_index = np.argmax(similarities)
    return documents[best_index], similarities[best_index]


"""Phase 8: Keyword-Based Search Implementation"""
translator = str.maketrans("", "", string.punctuation)

def keyword_search(query):
    query_words = set(query.lower().translate(translator).split())
    scores = []

    for doc in documents:
        doc_words = set(doc.lower().translate(translator).split())
        match_count = len(query_words.intersection(doc_words))
        scores.append(match_count)

    best_index = np.argmax(scores)
    return documents[best_index], scores[best_index]


query = input("Enter your search query:\n")

keyword_doc, keyword_score = keyword_search(query)
semantic_doc, semantic_score = semantic_search(query)

print("\nSemantic Search Result:")
print("Document:", semantic_doc)
print(f"Similarity Score: {semantic_score:.4f}")

print("\nKeyword Search Result:")
print("Document:", keyword_doc)
print("Keyword Match Score:", keyword_score)


"""Phase 9: Augmenting LLM Prompts with Semantic Search Results"""
augmentedPrompt = f"""
You are an AI assistant.

Context: {semantic_doc}
User Query: {query}

Answer the user's query based on the provided context.
"""
print("\nAugmented Prompt for LLM:", augmentedPrompt)


"""Phase 10: Conceptual Reflection:
Embeddings convert text into dense numerical vectors that capture semantic meaning rather than just exact words. Words or sentences with similar meanings 
are represented by vectors that are close together in mathematical space. This allows systems to measure meaning using similarity metrics such as cosine 
similarity. Unlike keyword matching, embeddings understand synonyms, paraphrasing, and contextual relationships. Semantic search uses these embeddings to retrieve 
information based on meaning rather than exact word overlap. This is critical for GenAI systems because it enables accurate document retrieval, better context 
selection, and reduced hallucinations. Without embeddings and semantic search, modern systems like RAG would not be able to provide context-aware and intelligent responses."""