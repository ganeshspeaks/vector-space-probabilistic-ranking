def precision_at_k(ranked_docs, relevant_docs, k=10):
    if not ranked_docs or k == 0:
        return 0.0

    top_k = ranked_docs[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs)

    return relevant_in_top_k / k


def evaluate_queries(queries, qrels, rank_fn, k=10):
    per_query_scores = {}

    for qid, query_text in queries.items():
        if qid not in qrels:
            continue

        relevant_docs = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}

        if not relevant_docs:
            continue

        ranked_docs = rank_fn(query_text)

        p_at_k = precision_at_k(ranked_docs, relevant_docs, k)
        per_query_scores[qid] = p_at_k

    if not per_query_scores:
        return 0.0, {}

    avg_precision = sum(per_query_scores.values()) / len(per_query_scores)

    return avg_precision, per_query_scores
