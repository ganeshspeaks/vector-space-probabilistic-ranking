import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_tfidf_matrix(docs):
    doc_ids = list(docs.keys())
    doc_texts = [docs[doc_id] for doc_id in doc_ids]

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X = vectorizer.fit_transform(doc_texts)

    return vectorizer, X, doc_ids


def rank_documents(query, vectorizer, X, doc_ids):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()

    ranked_indices = np.argsort(similarities)[::-1]
    ranked_doc_ids = [doc_ids[i] for i in ranked_indices]
    scores = similarities[ranked_indices]

    return ranked_doc_ids, scores
