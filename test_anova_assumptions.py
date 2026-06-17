import sys
import requests
import pandas as pd
import io
import json

API_URL = "http://127.0.0.1:8000"

def test_twofactor_assumptions():
    print("--- Starting Two-Factor ANOVA Assumptions Verification ---")

    # Login
    login_data = {
        "username": "ravi.scholar@icar-iiss.res.in",
        "password": "PhDpassword2026!"
    }
    r = requests.post(f"{API_URL}/auth/login", data=login_data)
    if r.status_code != 200:
        print("Login failed.")
        sys.exit(1)
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Load test data
    df = pd.read_csv("../test_anova.csv")
    df['Block'] = 'Block' + (df.groupby(['Fertilizer', 'Water']).cumcount() + 1).astype(str)

    # Helper
    def send_anova_request(test_type, df_input, dep_var, ind_var1, ind_var2=None, rep_var=None):
        csv_buf = io.StringIO()
        df_input.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        files = {"file": ("test_anova.csv", csv_buf.getvalue(), "text/csv")}
        data = {
            "test_type": test_type,
            "dep_var": dep_var,
            "ind_var1": ind_var1,
            "ind_var2": ind_var2
        }
        if rep_var:
            data["rep_var"] = rep_var
        r = requests.post(f"{API_URL}/analyze/anova", headers=headers, files=files, data=data)
        return r

    # 1. Test CRD Two-Way
    print("\n[Test 1] Testing CRD Two-Way...")
    res = send_anova_request("twoway", df, "Yield", "Fertilizer", ind_var2="Water")
    assert res.status_code == 200
    res_data = res.json()
    print("Shapiro Results:", json.dumps(res_data.get("shapiro_results"), indent=2))
    print("Levene Results:", json.dumps(res_data.get("levene_results"), indent=2))
    assert "Model Residuals" in res_data.get("shapiro_results", {})
    assert "stat" in res_data["shapiro_results"]["Model Residuals"]
    assert "p_value" in res_data["shapiro_results"]["Model Residuals"]
    assert res_data["levene_results"] is not None
    assert "stat" in res_data["levene_results"]
    print("CRD Two-Way Assumptions Checked successfully.")

    # 2. Test RBD Two-Way
    print("\n[Test 2] Testing RBD Two-Way...")
    res = send_anova_request("rbd_twoway", df, "Yield", "Fertilizer", ind_var2="Water", rep_var="Block")
    assert res.status_code == 200
    res_data = res.json()
    print("Shapiro Results:", json.dumps(res_data.get("shapiro_results"), indent=2))
    print("Levene Results:", json.dumps(res_data.get("levene_results"), indent=2))
    assert "Model Residuals" in res_data.get("shapiro_results", {})
    assert res_data["levene_results"] is not None
    print("RBD Two-Way Assumptions Checked successfully.")

    # 3. Test Split-plot Design
    print("\n[Test 3] Testing Split-plot...")
    res = send_anova_request("splitplot", df, "Yield", "Fertilizer", ind_var2="Water", rep_var="Block")
    assert res.status_code == 200
    res_data = res.json()
    print("Shapiro Results:", json.dumps(res_data.get("shapiro_results"), indent=2))
    print("Levene Results:", json.dumps(res_data.get("levene_results"), indent=2))
    assert "Model Residuals" in res_data.get("shapiro_results", {})
    assert res_data["levene_results"] is not None
    print("Split-plot Assumptions Checked successfully.")

    print("\n--- All Two-Factor ANOVA Assumptions Verification Tests Passed! ---")

if __name__ == "__main__":
    test_twofactor_assumptions()
