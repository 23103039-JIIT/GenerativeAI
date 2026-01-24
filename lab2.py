import string
import math
from collections import Counter, defaultdict

"""Phase 1: Raw Text Input & Sequential Representation"""
text = input("Input text: ")
# Store the text as sequence (list of words)
wordTokens = text.split() 
# Print each word along with its position index
for i, word in enumerate(wordTokens):
    print(f"Index {i}: {word}")


"""Phase 2: Tokenization Pipeline"""
# Character Level Tokenization
charTokens = list(text)
print("Tokenization: ")
# Token Statistics
print(f"Word tokens: {wordTokens}")
print(f"Character tokens: {charTokens}")
print("Number of word tokens = ", len(wordTokens))
print("Number of character tokens = ", len(charTokens))


"""Phase 3: Token Normalization & Frequency Analysis"""
# Convert text to lowercase
text = text.lower() 
# Delete punctuation from string
translator = str.maketrans("", "", string.punctuation) 
# Clean Text
cleanText = text.translate(translator) 
# Word level tokenization using clean text
WordTokens = cleanText.split()
# Character level tokenization
CharTokens = list(cleanText)
# Count frequency of each word token
WordFrequency = Counter(WordTokens) 
print("\n--- Top 10 Most Frequent Word Tokens ---")
for word, freq in WordFrequency.most_common(10):
    print(f"{word}: {freq}")
print("\n--- Token Statistics ---")
print("Total word tokens:", len(WordTokens))
print("Total character tokens:", len(CharTokens))


"""Phase 4: Simple Next-Token Prediction"""
# Build bigram dictionary --> bigram[word] = list of words that follow it
bigram = defaultdict(list)
for i in range(len(WordTokens) - 1):
    currentWord = WordTokens[i]
    nextWord = WordTokens[i+1]
    bigram[currentWord].append(nextWord)
# User input for prediction
query = input("Enter word to predict next token: ").lower()
# Predict most likely next word
if query in bigram:
    nextPossible = bigram[query]
    mostCommonNext = Counter(nextPossible).most_common(1)[0][0]
    print(f"Next predicted word after {query} = {mostCommonNext}")
else:
    print(f"No prediction available for {query}")


"""Phase 5: Creating Simple Word Embeddings"""
# Create vocabulary for unique words
vocabulary = sorted(set(WordTokens))
vocab_size = len(vocabulary)
# Assign index to each word
wordIndex = {word : index for index, word in enumerate(vocabulary)}
# Create one-hot embeddings
embeddings = {}
for word, index in wordIndex.items():
    vector = [0] * vocab_size
    vector[index] = 1
    embeddings[word] = vector

# Display embeddings for sample words
print("\nSample Word Embeddings (One-Hot)")
sample_words = vocabulary[:5]   # display first 5 words
for word in sample_words:
    print(f"{word}: {embeddings[word]}")


"""Phase 6: Word Similarity Using Embeddings"""
# Function to compute cosine similarity
def cosine_similarity(v1, v2):
    dotProduct = sum(a * b for a,b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0
    return dotProduct / (mag1 * mag2)

# Accept 2 words from user
w1 = input("Enter 1st word: ").lower()
w2 = input("Enter 2nd word: ").lower()

# Compute similarity
if w1 in embeddings and w2 in embeddings:
    similarity = cosine_similarity(embeddings[w1], embeddings[w2])
    print(f"Cosine similarity b/w '{w1}' and '{w2}' = {similarity:.2f}")
    # Decide similarity using threshold
    threshold = 0.5
    if similarity >= threshold:
        print("The words are similar.")
    else:
        print("The words are distinct")

else:
    print("One or both words are not present in the vocabulary.")


"""Phase 7: Sentence Embedding & Similarity"""
# Function to compute sentence embeddings
def sentence_embeddings(sentence):
    sentence = sentence.lower().translate(translator)
    words = sentence.split()
    vectors = [embeddings[word] for word in words if word in embeddings]

    if not vectors:
        return [0] * vocab_size
    
    avgVector = [sum(values) / len(vectors) for values in zip(*vectors)]
    return avgVector

# Accept two sentences
s1 = input("Enter first sentence: ")
s2 = input("Enter second sentence: ")

# Compute sentence embeddings & similarity
v1 = sentence_embeddings(s1)
v2 = sentence_embeddings(s2)
similarity = cosine_similarity(v1, v2)
print(f"Sentence similarity score: {similarity:.2f}")


"""Phase 10: Conceptual Reflection

Raw text is first collected as a sequence of characters and words, preserving order.
It is then tokenized into smaller units such as words or characters to make processing manageable.
Normalization steps like lowercasing and punctuation removal ensure consistency in representation.
Tokens are converted into numeric vectors (embeddings) so machines can perform mathematical operations.
These vectors enable similarity measurement, prediction, and retrieval using algorithms.
Such transformations are essential because Generative AI models operate on numbers, not text.
Without this pipeline, models would be unable to learn patterns, meaning, or context from language.

"""