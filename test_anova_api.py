import sys
import requests
import pandas as pd
import io
import json

API_URL = "http://127.0.0.1:8000"

def run_tests():
    print("--- Starting ANOVA API Verification ---")

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

    # Load test data
    df = pd.read_csv("../test_anova.csv")
    # Add block column dynamically for block designs (1 to 5)
    df['Block'] = 'Block' + (df.groupby(['Fertilizer', 'Water']).cumcount() + 1).astype(str)
    print("Loaded test_anova.csv and generated 'Block' column.")

    # Helper function to send anova requests
    def send_anova_request(test_type, df_input, dep_var, ind_var1, ind_var2=None, rep_var=None):
        csv_buf = io.StringIO()
        df_input.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        
        files = {"file": ("test_anova.csv", csv_buf.getvalue(), "text/csv")}
        data = {
            "test_type": test_type,
            "dep_var": dep_var,
            "ind_var1": ind_var1
        }
        if ind_var2:
            data["ind_var2"] = ind_var2
        if rep_var:
            data["rep_var"] = rep_var

        r = requests.post(f"{API_URL}/analyze/anova", headers=headers, files=files, data=data)
        return r

    # Test 1: One-Factor CRD (oneway)
    print("\nTesting One-Factor CRD (oneway)...")
    res = send_anova_request("oneway", df, "Yield", "Fertilizer")
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    res_data = res.json()
    assert "anova_table" in res_data
    assert "cd_results" in res_data
    print("One-Factor CRD Success!")
    print(f"CD Results: {json.dumps(res_data['cd_results'], indent=2)}")

    # Test 2: One-Factor RBD (rbd_oneway)
    # Filter to Water == 'Rainfed' to have 1 observation per treatment-block cell
    df_rbd_oneway = df[df['Water'] == 'Rainfed']
    print("\nTesting One-Factor RBD (rbd_oneway)...")
    res = send_anova_request("rbd_oneway", df_rbd_oneway, "Yield", "Fertilizer", rep_var="Block")
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    res_data = res.json()
    assert "anova_table" in res_data
    assert "cd_results" in res_data
    assert "df_rep" in res_data["anova_table"]
    print("One-Factor RBD Success!")
    print(f"CD Results: {json.dumps(res_data['cd_results'], indent=2)}")

    # Test 3: Two-Factor CRD (twoway)
    print("\nTesting Two-Factor CRD (twoway)...")
    res = send_anova_request("twoway", df, "Yield", "Fertilizer", ind_var2="Water")
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    res_data = res.json()
    assert "anova_table" in res_data
    assert "cd_results" in res_data
    assert "factorA" in res_data["anova_table"]
    assert "factorB" in res_data["anova_table"]
    assert "interaction" in res_data["anova_table"]
    print("Two-Factor CRD Success!")
    print(f"CD Results: {json.dumps(res_data['cd_results'], indent=2)}")

    # Test 4: Two-Factor RBD (rbd_twoway)
    print("\nTesting Two-Factor RBD (rbd_twoway)...")
    res = send_anova_request("rbd_twoway", df, "Yield", "Fertilizer", ind_var2="Water", rep_var="Block")
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    res_data = res.json()
    assert "anova_table" in res_data
    assert "cd_results" in res_data
    assert "df_rep" in res_data["anova_table"]
    assert "factorA" in res_data["anova_table"]
    assert "interaction" in res_data["anova_table"]
    print("Two-Factor RBD Success!")
    print(f"CD Results: {json.dumps(res_data['cd_results'], indent=2)}")

    # Test 5: Split-plot Design (splitplot)
    print("\nTesting Split-plot Design (splitplot)...")
    res = send_anova_request("splitplot", df, "Yield", "Fertilizer", ind_var2="Water", rep_var="Block")
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    res_data = res.json()
    assert "anova_table" in res_data
    assert "cd_results" in res_data
    assert "error_a" in res_data["anova_table"]
    assert "error_b" in res_data["anova_table"]
    assert len(res_data["cd_results"]) == 4 # A, B, B at same A, A at same B
    print("Split-plot Design Success!")
    print(f"CD Results: {json.dumps(res_data['cd_results'], indent=2)}")

    print("\n--- All ANOVA API Verification Tests Passed! ---")

if __name__ == "__main__":
    run_tests()
