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

# Compute embeddings for all documents
docVectors = [sentence_embeddings(doc) for doc in documents]

# Accept user query
query = input("Enter query sentence: ")
queryVector = sentence_embeddings(query)

# Retrieve most similar document
similarities = [cosine_similarity(queryVector, docVc) for docVc in docVectors]
bestMatchIndex = similarities.index(max(similarities))
print("\nMost Similar Document:")
print(documents[bestMatchIndex])
print(f"Similarity Score: {similarities[bestMatchIndex]:.2f}")