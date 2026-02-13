import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import json

def main():
    print("🧠 Starting speaker clustering...")
    
    # 1. Load embeddings
    try:
        with open("embeddings.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("❌ Could not find 'embeddings.pkl'. Run extract_embeddings.py first!")
        return

    name_list = list(data.keys())
    embeddings = list(data.values())
    X = np.array(embeddings)
    
    print(f"📊 Analyzing {len(name_list)} unique speaker clips...")

    # 2. Setup Clustering
    clusterer = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.3, 
        metric="cosine", 
        linkage="average"
    )
    
    # 3. Predict clusters
    labels = clusterer.fit_predict(X)
    
    # 4. Organize results
    results = {}
    for name, cluster_id in zip(name_list, labels):
        global_id = f"GLOBAL_SPEAKER_{cluster_id}"
        if global_id not in results:
            results[global_id] = []
        results[global_id].append(name)

    # 5. Display results
    print("\n" + "="*40)
    print("🏆 CLUSTERING RESULTS 🏆")
    print("="*40)
    
    sorted_results = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
    
    for global_id, episodes in sorted_results:
        print(f"\n{global_id} (Appeared in {len(episodes)} episodes):")
        for ep in episodes[:5]:
            print(f"  - {ep}")
        if len(episodes) > 5:
            print(f"  - ... plus {len(episodes)-5} more.")

    # Save mapping
    with open("global_speakers.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n✅ Done! Mapping saved to 'global_speakers.json'.")

if __name__ == "__main__":
    main()
