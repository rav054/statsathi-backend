import sys
import requests
import pandas as pd
import io

API_URL = "http://127.0.0.1:8000"

def test_clustering():
    print("--- Starting Clustering API Verification ---")

    # 1. Health check
    try:
        r = requests.get(f"{API_URL}/")
        print(f"Health check status: {r.status_code}")
        assert r.status_code == 200
    except Exception as e:
        print(f"Backend is not accessible: {e}")
        sys.exit(1)

    # 2. Login or Sign Up
    signup_payload = {
        "email": "ravi.scholar@icar-iiss.res.in",
        "full_name": "Dr. Ravi Kumar",
        "institution": "ICAR-IISS Bhopal",
        "password": "PhDpassword2026!"
    }
    r = requests.post(f"{API_URL}/auth/signup", json=signup_payload)
    if r.status_code == 400 and "already registered" in r.json().get("detail", ""):
        print("User already registered, continuing to login...")
    else:
        assert r.status_code == 201
        print("User signed up successfully!")

    login_data = {
        "username": "ravi.scholar@icar-iiss.res.in",
        "password": "PhDpassword2026!"
    }
    r = requests.post(f"{API_URL}/auth/login", data=login_data)
    assert r.status_code == 200
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("Logged in successfully.")

    # 3. Test K-Means Clustering
    cluster_url = f"{API_URL}/analyze/cluster"
    
    # Open test1.csv file
    files = {
        "file": ("test1.csv", open("../test1.csv", "rb"), "text/csv")
    }
    data = {
        "variables_str": "Mn PPM,Zn PPM,Cd ppm,Cr ppm",
        "method": "kmeans",
        "k": 3
    }

    print("\nTesting K-Means Clustering...")
    r = requests.post(cluster_url, headers=headers, files=files, data=data)
    print(f"K-Means Response Status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        print("K-Means clustering succeeded!")
        
        # Verify K-Means keys
        assert "kmeans" in res
        kmeans_res = res["kmeans"]
        print(f"  Number of centroids: {len(kmeans_res['centroids'])}")
        print(f"  Number of labels: {len(kmeans_res['labels'])}")
        print(f"  Summaries count: {len(kmeans_res['summaries'])}")
        assert len(kmeans_res['labels']) > 0
        assert len(kmeans_res['centroids']) == 3

        # Verify PCA keys
        assert "pca" in res
        pca_res = res["pca"]
        print(f"  PC1/PC2 Explained Variance Ratio: {pca_res['explained_variance']}")
        
        # Verify Data Table output
        assert "records" in res
        print(f"  Total observations in data: {len(res['records'])}")
    else:
        print(f"K-Means Error: {r.text}")
        sys.exit(1)

    # 4. Test Hierarchical Clustering
    # Re-open file because it might be closed/consumed in python requests
    files = {
        "file": ("test1.csv", open("../test1.csv", "rb"), "text/csv")
    }
    data = {
        "variables_str": "Mn PPM,Zn PPM,Cd ppm,Cr ppm",
        "method": "hierarchical",
        "k": 3,
        "cut_height": 10.5
    }

    print("\nTesting Hierarchical Clustering...")
    r = requests.post(cluster_url, headers=headers, files=files, data=data)
    print(f"Hierarchical Response Status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        print("Hierarchical clustering succeeded!")
        
        # Verify Hierarchical keys
        assert "hierarchical" in res
        h_res = res["hierarchical"]
        print("  Dendrogram keys: ", h_res["dendrogram"].keys())
        print(f"  Number of labels: {len(h_res['labels'])}")
        print(f"  Summaries count: {len(h_res['summaries'])}")
        assert len(h_res['labels']) > 0
        assert "icoord" in h_res["dendrogram"]
        assert "dcoord" in h_res["dendrogram"]
    else:
        print(f"Hierarchical Error: {r.text}")
        sys.exit(1)

    print("\nAll Clustering API Verification Tests Passed!")

if __name__ == "__main__":
    test_clustering()
