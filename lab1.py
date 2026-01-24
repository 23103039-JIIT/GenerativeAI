import string 
import math
from collections import Counter, defaultdict

"""Phase 8: Similarity-Based Document Retrieval"""

# Store multiple documents
documents = [
    "Artificial intelligence is transforming technology",
    "Machine learning enables systems to learn from data",
    "AI is used in healthcare and finance",
    "Natural language processing deals with text data"
]

# Build corpus from documents
translator = str.maketrans("", "", string.punctuation)
corpus = " ".join(documents).lower().translate(translator)
tokens = corpus.split()

# Create vocabulary & one-hot embeddings
vocabulary = sorted(set(tokens))
vocab_size = len(vocabulary)

wordIndex = {word: index for index, word in enumerate(vocabulary)}
embeddings = {}
for word, index in wordIndex.items():
    vector = [0] * vocab_size
    vector[index] = 1
    embeddings[word] = vector

# Sentence embedding function
def sentence_embeddings(sentence):
    sentence = sentence.lower().translate(translator)
    words = sentence.split()
    vectors = [embeddings[word] for word in words if word in embeddings]

    if not vectors:
        return [0] * vocab_size
    
    avgVector = [sum(values) / len(vectors) for values in zip(*vectors)]
    return avgVector

# Function to compute cosine similarity
def cosine_similarity(v1, v2):
    dotProduct = sum(a * b for a,b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0
    return dotProduct / (mag1 * mag2)

# Keyword-based retrieval
def keyword_retrieval(query):
    query_words = set(query.lower().translate(translator).split())
    scores = []

    for doc in documents:
        doc_words = set(doc.lower().translate(translator).split())
        match_count = len(query_words.intersection(doc_words))
        scores.append(match_count)
    
    best_index = scores.index(max(scores))
    return documents[best_index], scores[best_index]

# Compute embeddings for all documents
docVectors = [sentence_embeddings(doc) for doc in documents]

# Similarity-based retrieval
def similarity_retrieval(query):
    query_vec = sentence_embeddings(query)
    similarities = [cosine_similarity(query_vec, doc_vec) for doc_vec in docVectors]
    best_index = similarities.index(max(similarities))
    return documents[best_index], similarities[best_index]

# Accept user query
query = input("Enter query sentence: ")
keyword_doc, keyword_score = keyword_retrieval(query)
similar_doc, similarity_score = similarity_retrieval(query)

# Comparison
print("\n--- Retrieval Comparison ---")

print("\nKeyword-Based Retrieval:")
print("Retrieved Document:", keyword_doc)
print("Keyword Match Score:", keyword_score)

print("\nSimilarity-Based Retrieval:")
print("Retrieved Document:", similar_doc)
print(f"Similarity Score: {similarity_score:.2f}")