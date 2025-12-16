import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


CLEAN_QUERY = "genetic research found a cure for cancer"

DOCUMENTS = {
    "D1": "g enetic r esearch found a c ure for cancer new",
    "D2": "chemotherapy treatment is still common in most tumors",
    "D3": "fast blood tests may detect cancer early",
    "D4": "developing new drugs requires many clinical trials",
    "D5": "Dr. Levi developed an experimental c ure",
    "D6": "c ancer is a collective name for a group of diseases",
    "D7": "research genetic in animals brought hope",
    "D8": "the success of the c ure was measured by survival",
    "D9": "immunotherapy changes the rules of the game",
    "D10": "g enetics and immunology combine together",
    "D11": "radiation therapy is one of the oldest tools",
    "D12": "this c ure works by changing DNA",
    "D13": "breast c ancer is the most common in women",
    "D14": "new research talks about targeted treatment",
    "D15": "scientists found a way to prevent the disease",
    "D16": "the immune system is the key to recovery",
    "D17": "current r esearch focuses on cellular changes",
    "D18": "this expensive c u r e is not yet approved",
    "D19": "proper nutrition may reduce the risk of c ancer",
    "D20": "genetic advanced research gives hope",
}

DOC_IDS = list(DOCUMENTS.keys())


qgram_vectorizer = CountVectorizer(
    analyzer='char',  # Use character-level analysis
    ngram_range=(3, 3)  # Use 3-grams (q=3)
)

document_qgram_vectors = qgram_vectorizer.fit_transform(DOCUMENTS.values())

query_qgram_vector = qgram_vectorizer.transform([CLEAN_QUERY])

document_vectors_np = document_qgram_vectors.toarray()
query_vector_np = query_qgram_vector.toarray()

print(f"Total number of unique 3-grams (Vocabulary Size): {document_vectors_np.shape[1]}")
print(f"Document Matrix Shape: {document_vectors_np.shape}")
print("-" * 100)


similarity_scores = cosine_similarity(query_vector_np, document_vectors_np)

scores = similarity_scores[0]

ranking = list(zip(DOC_IDS, scores))

ranking.sort(key=lambda x: x[1], reverse=True)


print(" Information Retrieval Results (Ranked by Cosine Similarity using 3-grams):")
print(f"Clean Query: '{CLEAN_QUERY}'")
print("-" * 120)
print(f"{'Rank':<5} {'Doc ID':<8} {'Similarity Score':<20} {'Document Text (with Simulated OCR Errors)':<80}")
print("-" * 120)

for i, (doc_id, score) in enumerate(ranking):
    original_text = DOCUMENTS[doc_id]

    print(f"{i + 1:<5} {doc_id:<8} {score:<20.4f} {original_text:<80}")

print("-" * 120)

print("\n Analysis:")
print("The high scores for documents like D1, D7, and D15 demonstrate the success of 3-gram indexing.")
print("Even though query terms are corrupted in the documents (e.g., 'research' vs 'r esearch' in D1),")
print(
    "the overlapping character 3-grams (e.g., 'res', 'sea', 'ear' from 'research' and 'r e', ' es', 'sea', 'ear' from 'r esearch')")
print(
    "ensure a significant Cosine Similarity score, allowing the system to retrieve the relevant document despite the OCR noise.")
print("")