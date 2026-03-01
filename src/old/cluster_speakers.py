import os
import boto3
import pickle
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def main():
    BUCKET = os.getenv("S3_BUCKET", "ml-data")
    BASE_PATH = os.getenv("S3_BASE_PATH", "002_speech_dataset")
    EMBEDDINGS_KEY = f"{BASE_PATH}/metadata/embeddings.pkl"
    RESULTS_KEY = f"{BASE_PATH}/metadata/global_speakers.json"
    
    print("🧠 Starter klynging/sortering av stemmer...")
    
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # 1. Last inn fingeravtrykkene direkte fra S3
    try:
        print(f"📥 Laster ned {EMBEDDINGS_KEY} fra S3...")
        response = s3.get_object(Bucket=BUCKET, Key=EMBEDDINGS_KEY)
        data = pickle.loads(response['Body'].read())
    except Exception as e:
        print(f"❌ Fant ikke '{EMBEDDINGS_KEY}' i S3. Kjør Jobb A først!")
        return

    name_list = list(data.keys())
    embeddings = list(data.values())
    X = np.array(embeddings)
    
    print(f"📊 Analyserer {len(name_list)} unike stemmeklipp...")

    # 2. Sett opp Clustering
    clusterer = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.3, 
        metric="cosine", 
        linkage="average"
    )
    
    # 3. La AI gjette
    labels = clusterer.fit_predict(X)
    
    # 4. Organiser resultatene
    results = {}
    for name, cluster_id in zip(name_list, labels):
        global_id = f"GLOBAL_SPEAKER_{cluster_id}"
        if global_id not in results:
            results[global_id] = []
        results[global_id].append(name)

    # 5. Vis resultatene i loggen
    print("\n" + "="*40)
    print("🏆 SORTERINGSRESULTAT 🏆")
    print("="*40)
    
    sorted_results = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
    
    for global_id, episodes in sorted_results:
        print(f"\n{global_id} (Dukket opp i {len(episodes)} episoder):")
        for ep in episodes[:5]:
            print(f"  - {ep}")
        if len(episodes) > 5:
            print(f"  - ... pluss {len(episodes)-5} flere.")

    # 6. Lagre resultatene tilbake til S3
    print(f"\n📤 Lagrer fasiten til S3: {RESULTS_KEY}")
    s3.put_object(
        Bucket=BUCKET,
        Key=RESULTS_KEY,
        Body=json.dumps(results, indent=4),
        ContentType="application/json"
    )
        
    print("\n✅ Jobb B er ferdig! Du kan nå laste ned 'global_speakers.json' fra S3-bøtten din.")

if __name__ == "__main__":
    main()
