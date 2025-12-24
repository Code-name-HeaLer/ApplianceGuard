import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from src.features.extraction import extract_features

def build_graph_pipeline(all_signals, train_indices, n_clusters=9, seed=42):
    """
    Orchestrates the graph construction: Feature Extraction -> Clustering -> Graph.
    
    Args:
        all_signals (np.ndarray): [N, seq_len] array of signals.
        train_indices (list): Indices to use for fitting KMeans.
        n_clusters (int): Number of nodes in the graph.
        
    Returns:
        dict: A dictionary containing the graph and all artifacts.
    """
    print("   [GraphBuilder] Extracting features...")
    features_list = [extract_features(win) for win in all_signals]
    # Filter Nones if any
    features_list = [f if f is not None else np.zeros(15) for f in features_list]
    all_features = np.array(features_list)

    # Scale Features (Fit on Train only)
    print("   [GraphBuilder] Scaling features...")
    feature_scaler = StandardScaler()
    feature_scaler.fit(all_features[train_indices])
    scaled_features = feature_scaler.transform(all_features)
    scaled_features = np.nan_to_num(scaled_features)

    # Clustering
    print(f"   [GraphBuilder] Clustering into {n_clusters} nodes...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    train_features = scaled_features[train_indices]
    
    # Fit on train, predict on ALL
    kmeans.fit(train_features)
    all_labels = kmeans.predict(scaled_features)
    feature_centroids = kmeans.cluster_centers_

    # Calculate Node Features (Average Signal per Cluster)
    print("   [GraphBuilder] Calculating node embeddings...")
    seq_len = all_signals.shape[1]
    node_features = np.zeros((n_clusters, seq_len), dtype=np.float32)
    counts = np.zeros(n_clusters)

    # Map train indices to labels
    train_labels = all_labels[train_indices]
    
    for i, original_idx in enumerate(train_indices):
        label = train_labels[i]
        node_features[label] += all_signals[original_idx]
        counts[label] += 1
        
    # Average
    for k in range(n_clusters):
        if counts[k] > 0:
            node_features[k] /= counts[k]

    # Transition Matrix (Adjacency)
    print("   [GraphBuilder] Calculating transition matrix...")
    transition_counts = np.zeros((n_clusters, n_clusters), dtype=int)
    
    # Count transitions in training data sequence
    for i in range(len(train_labels) - 1):
        curr = train_labels[i]
        next_ = train_labels[i+1]
        transition_counts[curr, next_] += 1
        
    # Laplace Smoothing & Normalization
    transition_counts += 1 
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = transition_counts / row_sums

    # Build NetworkX Graph
    G = nx.DiGraph()
    for i in range(n_clusters):
        G.add_node(i, features=node_features[i])
        
    for i in range(n_clusters):
        for j in range(n_clusters):
            weight = transition_probs[i, j]
            if weight > 1e-6:
                G.add_edge(i, j, weight=float(weight))

    return {
        "graph": G,
        "feature_centroids": feature_centroids,
        "node_features": node_features,
        "transition_matrix": transition_probs,
        "feature_scaler": feature_scaler,
        "all_labels": all_labels
    }