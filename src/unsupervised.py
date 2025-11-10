import os, numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from preprocess import load_data
import joblib

def run_unsupervised():
    X_train, X_test, Y_train, Y_test, scaler = load_data()
    X_full = np.vstack((X_train, X_test))

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_full)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(models_dir, "clustering_model.pkl"))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_full)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA + KMeans Clustering")
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "clusters_pca.png")) # save figure
    plt.show()
    print("Unsupervised model and figure saved!")

if __name__ == "__main__":
    run_unsupervised()
