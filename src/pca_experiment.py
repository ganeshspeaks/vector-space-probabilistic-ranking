import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def compute_pca_variance(X_dense, k_values=[20, 50, 100]):
    n_samples = X_dense.shape[0]
    n_components = min(n_samples, X_dense.shape[1])

    pca_full = PCA(n_components=n_components)
    pca_full.fit(X_dense)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    variances = {}
    for k in k_values:
        if k <= len(cumulative_variance):
            variances[k] = cumulative_variance[k - 1]
        else:
            variances[k] = cumulative_variance[-1]

    return variances, pca_full, cumulative_variance


def reduce_dimensions(X_dense, n_components=100):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_dense)

    return X_reduced, pca


def create_scree_plot(cumulative_variance, output_path="results/scree_plot.png"):
    plt.figure(figsize=(10, 6))

    n_components = len(cumulative_variance)
    plt.plot(range(1, n_components + 1), cumulative_variance, "b-", linewidth=2)
    plt.axhline(y=0.9, color="r", linestyle="--", label="90% variance")

    for k in [20, 50, 100]:
        if k <= n_components:
            plt.axvline(x=k, color="g", linestyle=":", alpha=0.5)
            plt.plot(k, cumulative_variance[k - 1], "ro", markersize=8)
            plt.annotate(
                f"k={k}\n{cumulative_variance[k - 1]:.2%}",
                xy=(k, cumulative_variance[k - 1]),
                xytext=(k + 20, cumulative_variance[k - 1] - 0.05),
                fontsize=9,
            )

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance (Cranfield Dataset)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Scree plot saved to {output_path}")
