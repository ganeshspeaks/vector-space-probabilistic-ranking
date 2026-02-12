import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from evaluation import evaluate_queries
from load_data import load_cranfield
from pca_experiment import compute_pca_variance, create_scree_plot, reduce_dimensions
from vector_space import create_tfidf_matrix, rank_documents


def main():
    print("=" * 60)
    print("Vector Space Probabilistic Ranking Experiment")
    print("=" * 60)

    # Loading data
    print("\n[1] Loading Cranfield dataset...")
    docs, queries, qrels = load_cranfield()
    print(f"  Documents: {len(docs)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels: {len(qrels)}")

    # Creating TF-IDF matrix
    print("\n[2] Creating TF-IDF matrix...")
    vectorizer, X_tfidf, doc_ids = create_tfidf_matrix(docs)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  TF-IDF matrix shape: {X_tfidf.shape}")

    # TF-IDF Ranking and Evaluation
    print("\n[3] Evaluating TF-IDF ranking (Precision@10)...")

    def rank_fn_tfidf(query_text):
        ranked_docs, _ = rank_documents(query_text, vectorizer, X_tfidf, doc_ids)
        return ranked_docs

    avg_p10_tfidf, _ = evaluate_queries(queries, qrels, rank_fn_tfidf, k=10)
    print(f"  Average Precision@10 (TF-IDF): {avg_p10_tfidf:.4f}")

    # PCA Experiment
    print("\n[4] Running PCA experiment...")
    print("  Converting TF-IDF to dense matrix...")
    X_dense = X_tfidf.toarray()

    print("  Computing variance for k=20, 50, 100...")
    variances, pca_full, cumulative_variance = compute_pca_variance(
        X_dense, [20, 50, 100]
    )

    print("\n  Variance Preserved:")
    for k in [20, 50, 100]:
        print(f"    k={k}: {variances[k]:.2%}")

    # Creating scree plot
    print("\n  Creating scree plot...")
    os.makedirs("results", exist_ok=True)
    create_scree_plot(cumulative_variance, "results/scree_plot.png")

    # Reducing to 100 dimensions and evaluate
    print("\n[5] Evaluating PCA-100 ranking (Precision@10)...")
    X_pca100, pca_model = reduce_dimensions(X_dense, n_components=100)
    print(f"  Reduced matrix shape: {X_pca100.shape}")

    def rank_fn_pca(query_text):
        # Transforming query to TF-IDF, then to PCA space
        query_tfidf = vectorizer.transform([query_text]).toarray()
        query_pca = pca_model.transform(query_tfidf)

        # Computing cosine similarity in PCA space
        similarities = cosine_similarity(query_pca, X_pca100).flatten()

        # Rank documents
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_docs = [doc_ids[i] for i in ranked_indices]
        return ranked_docs

    avg_p10_pca, _ = evaluate_queries(queries, qrels, rank_fn_pca, k=10)
    print(f"  Average Precision@10 (PCA-100): {avg_p10_pca:.4f}")

    # Saving results
    print("\n[6] Saving results...")
    with open("results/metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Documents", len(docs)])
        writer.writerow(["Queries", len(queries)])
        writer.writerow(["Vocabulary Size", vocab_size])
        writer.writerow(["TF-IDF Precision@10", f"{avg_p10_tfidf:.4f}"])
        writer.writerow(["PCA-100 Precision@10", f"{avg_p10_pca:.4f}"])
        writer.writerow(["Variance at k=20", f"{variances[20]:.4f}"])
        writer.writerow(["Variance at k=50", f"{variances[50]:.4f}"])
        writer.writerow(["Variance at k=100", f"{variances[100]:.4f}"])

    print(f"  Results saved to results/metrics.csv")

    # Printing summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset Size: {len(docs)} documents, {len(queries)} queries")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"\nVariance Preserved:")
    print(f"  k=20:  {variances[20]:.2%}")
    print(f"  k=50:  {variances[50]:.2%}")
    print(f"  k=100: {variances[100]:.2%}")
    print(f"\nPrecision@10:")
    print(f"  TF-IDF:  {avg_p10_tfidf:.4f}")
    print(f"  PCA-100: {avg_p10_pca:.4f}")
    print(f"\nScree plot saved to: results/scree_plot.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
