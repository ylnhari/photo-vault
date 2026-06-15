import os
import json
import numpy as np
from sklearn.cluster import DBSCAN

FACE_DIR = r"C:\Users\ylnha\Projects\local-image-search\data\faces"
CLUSTERS_FILE = r"C:\Users\ylnha\Projects\local-image-search\data\person_map.json"

def cluster_faces(eps=0.5, min_samples=2):
    """
    Groups faces into clusters based on embedding similarity.
    """
    face_files = [f for f in os.listdir(FACE_DIR) if f.endswith('.json')]
    embeddings = []
    file_map = {}

    print(f"Loading {len(face_files)} face files...")
    for f in face_files:
        with open(os.path.join(FACE_DIR, f), 'r') as file:
            data = json.load(file)
            for item in data:
                embeddings.append(item['embedding'])
                file_map[len(embeddings)-1] = f

    if not embeddings:
        print("No faces found to cluster.")
        return

    print("Clustering faces...")
    X = np.array(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
    
    labels = db.labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[int(label)] = []
        clusters[int(label)].append(file_map[i])

    # Save clusters for manual labeling
    with open(CLUSTERS_FILE, 'w') as f:
        json.dump(clusters, f, indent=4)
    print(f"Clusters saved to {CLUSTERS_FILE}. Please label them manually.")
