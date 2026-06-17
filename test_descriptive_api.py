import sys
import requests
import pandas as pd
import io
import json

API_URL = "http://127.0.0.1:8000"

def run_tests():
    print("--- Starting Descriptive Statistics API Verification ---")

    # 1. Login to get token
    login_data = {
        "username": "ravi.scholar@icar-iiss.res.in",
        "password": "PhDpassword2026!"
    }
    try:
        r = requests.post(f"{API_URL}/auth/login", data=login_data)
        if r.status_code != 200:
            print("Failed to login. Please ensure the backend server is running and user is signed up.")
            sys.exit(1)
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("Logged in successfully.")
    except Exception as e:
        print(f"Error connecting to backend: {e}")
        sys.exit(1)

    # 2. Prepare test data
    # Create a small dataset with some missing values and an outlier to test everything
    data = {
        "Yield": [10.2, 11.5, 9.8, 12.1, 10.5, None, 11.0, 32.5, 9.9, 10.3], # 32.5 is a clear outlier
        "Treatment": ["T1", "T1", "T1", "T1", "T1", "T2", "T2", "T2", "T2", "T2"]
    }
    df = pd.DataFrame(data)
    
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    
    files = {"file": ("test_descriptive.csv", csv_buf.getvalue(), "text/csv")}
    payload = {
        "columns_str": "Yield",
        "group_var": "Treatment"
    }

    # 3. Call endpoint
    print("\nSending descriptive statistics request...")
    res = requests.post(f"{API_URL}/analyze/descriptive", headers=headers, files=files, data=payload)
    print(f"Status: {res.status_code}")
    
    if res.status_code != 200:
        print("Request failed:", res.text)
        sys.exit(1)
        
    res_data = res.json()
    print("Response parsed successfully.")
    
    # 4. Assert core calculations
    assert "variables" in res_data
    assert "Yield" in res_data["variables"]
    
    yield_data = res_data["variables"]["Yield"]
    assert "overall" in yield_data
    assert "groups" in yield_data
    
    overall = yield_data["overall"]
    print("\nOverall Yield Statistics:")
    print(f"Mean: {overall['mean']}")
    print(f"Median: {overall['median']}")
    print(f"CV%: {overall['cv']}%")
    print(f"Missing count: {overall['missing']['count']} ({overall['missing']['percentage']}%)")
    print(f"Outliers: {overall['outliers']}")
    
    assert overall["n"] == 9  # 10 rows minus 1 missing value
    assert overall["missing"]["count"] == 1
    assert overall["missing"]["percentage"] == 10.0
    
    # Verify outliers identified row index (32.5 is at pandas index 7, which is Excel row 9 because 0-index + 2)
    assert len(overall["outliers"]) == 1
    assert overall["outliers"][0]["row"] == 9
    assert overall["outliers"][0]["val"] == 32.5
    
    # Verify Shapiro and Q-Q
    assert "shapiro" in overall
    assert "stat" in overall["shapiro"]
    assert "qq" in overall
    assert "theoretical" in overall["qq"]
    assert len(overall["qq"]["theoretical"]) == 9
    
    # Verify Grouping
    groups = yield_data["groups"]
    assert "T1" in groups
    assert "T2" in groups
    
    print("\nGroup T1 Statistics:")
    print(f"Mean: {groups['T1']['mean']}")
    print(f"N: {groups['T1']['n']}")
    
    print("\nGroup T2 Statistics:")
    print(f"Mean: {groups['T2']['mean']}")
    print(f"N: {groups['T2']['n']}")
    print(f"Missing count: {groups['T2']['missing']['count']}")
    
    assert groups["T1"]["n"] == 5
    assert groups["T2"]["n"] == 4  # 5 rows minus 1 missing value
    assert groups["T2"]["missing"]["count"] == 1
    
    print("\n--- All Descriptive Statistics API Tests Passed! ---")

if __name__ == "__main__":
    run_tests()
