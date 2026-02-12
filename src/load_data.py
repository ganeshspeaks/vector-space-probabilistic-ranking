import ir_datasets


def load_cranfield():
    dataset = ir_datasets.load("cranfield")

    docs = {}
    for doc in dataset.docs_iter():
        docs[doc.doc_id] = doc.text

    queries = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text

    qrels = {}
    for qrel in dataset.qrels_iter():
        qid = qrel.query_id
        did = qrel.doc_id
        rel = qrel.relevance

        if qid not in qrels:
            qrels[qid] = {}

        qrels[qid][did] = rel

    return docs, queries, qrels
