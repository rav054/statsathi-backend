import sys
import requests
import pandas as pd
import io
import base64

API_URL = "http://127.0.0.1:8000"

def test_integration():
    print("--- Starting Integration Verification ---")
    
    # 1. Health check
    try:
      r = requests.get(f"{API_URL}/")
      print(f"Health check status: {r.status_code}, data: {r.json()}")
      assert r.status_code == 200
    except Exception as e:
      print(f"Backend is not accessible: {e}")
      sys.exit(1)

    # 2. User Sign Up
    signup_payload = {
        "email": "ravi.scholar@icar-iiss.res.in",
        "full_name": "Dr. Ravi Kumar",
        "institution": "ICAR-IISS Bhopal",
        "password": "PhDpassword2026!"
    }
    r = requests.post(f"{API_URL}/auth/signup", json=signup_payload)
    print(f"Signup response status: {r.status_code}")
    if r.status_code == 400 and "already registered" in r.json().get("detail", ""):
        print("User already registered, continuing to login...")
    else:
        assert r.status_code == 201
        print("User signed up successfully!")

    # 3. User Log In (OAuth2 password flow expects form data)
    login_data = {
        "username": "ravi.scholar@icar-iiss.res.in",
        "password": "PhDpassword2026!"
    }
    r = requests.post(f"{API_URL}/auth/login", data=login_data)
    print(f"Login response status: {r.status_code}")
    assert r.status_code == 200
    token_json = r.json()
    assert "access_token" in token_json
    token = token_json["access_token"]
    print("Logged in successfully, token retrieved!")

    # 4. Fetch User Profile (/auth/me)
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{API_URL}/auth/me", headers=headers)
    print(f"Profile check status: {r.status_code}, data: {r.json()}")
    assert r.status_code == 200
    assert r.json()["email"] == "ravi.scholar@icar-iiss.res.in"

    # 5. Generate mock scientific dataset and run Correlation Analysis
    data = {
        "Nitrogen_kg_ha": [60, 80, 100, 120, 140, 160],
        "Phosphorus_kg_ha": [30, 35, 40, 45, 50, 55],
        "CropYield_t_ha": [2.4, 2.9, 3.5, 4.1, 4.3, 4.5],
        "SoilType": ["Vertisol", "Vertisol", "Vertisol", "Vertisol", "Vertisol", "Vertisol"], # non-numeric
        "FertilizerType": ["Organic", "Chemical", "Organic", "Chemical", "Organic", "Chemical"],
        "YieldStatus": ["High", "Low", "High", "Low", "High", "Low"]
    }
    df = pd.DataFrame(data)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    # Call /analyze/correlation
    files = {"file": ("dataset_soil.csv", csv_buf.getvalue(), "text/csv")}
    r = requests.post(f"{API_URL}/analyze/correlation", headers=headers, files=files)
    print(f"Correlation response status: {r.status_code}")
    assert r.status_code == 200
    res_json = r.json()
    assert "plot" in res_json
    assert "matrix" in res_json
    
    plot_bytes = base64.b64decode(res_json["plot"])
    with open("test_heatmap_output.png", "wb") as f:
        f.write(plot_bytes)
    print("Success: Base64 correlation heatmap received and saved to 'test_heatmap_output.png'!")

    # 6. Test /analyze/columns
    csv_buf.seek(0)
    files_cols = {"file": ("dataset_soil.csv", csv_buf.getvalue(), "text/csv")}
    r = requests.post(f"{API_URL}/analyze/columns", headers=headers, files=files_cols)
    print(f"Columns response status: {r.status_code}")
    assert r.status_code == 200
    cols_data = r.json()
    assert "columns" in cols_data
    assert "numeric_columns" in cols_data
    print("Columns endpoint parsed headers successfully!")

    # 6.5. Test /analyze/preview
    csv_buf.seek(0)
    files_prev = {"file": ("dataset_soil.csv", csv_buf.getvalue(), "text/csv")}
    r = requests.post(f"{API_URL}/analyze/preview", headers=headers, files=files_prev)
    print(f"Preview response status: {r.status_code}")
    assert r.status_code == 200
    prev_data = r.json()
    assert "columns" in prev_data
    assert "data" in prev_data
    assert "total_rows" in prev_data
    assert len(prev_data["data"]) > 0
    print("Preview endpoint returned data records successfully!")

    # 7. Test /analyze/parametric
    csv_buf.seek(0)
    files_param = {"file": ("dataset_soil.csv", csv_buf.getvalue(), "text/csv")}
    param_data = {
        "test_type": "independent_t",
        "col1": "Nitrogen_kg_ha",
        "col2": "Phosphorus_kg_ha"
    }
    r = requests.post(f"{API_URL}/analyze/parametric", headers=headers, files=files_param, data=param_data)
    print(f"Parametric response status: {r.status_code}")
    assert r.status_code == 200
    param_res = r.json()
    assert "shapiro_results" in param_res
    assert "statistics" in param_res
    assert "plot" in param_res
    print("Parametric analysis independent t-test executed successfully!")
    print("Normality output:", param_res["shapiro_results"])
    print("Statistics output:", param_res["statistics"])
    
    # Save plot
    plot_bytes = base64.b64decode(param_res["plot"])
    with open("test_parametric_plot.png", "wb") as f:
        f.write(plot_bytes)
    print("Success: Base64 parametric comparison chart saved to 'test_parametric_plot.png'!")

    # 8. Test /analyze/nonparametric (Mann-Whitney U Test)
    csv_buf.seek(0)
    files_nonparam = {"file": ("dataset_soil.csv", csv_buf.getvalue(), "text/csv")}
    nonparam_data = {
        "test_type": "mann_whitney",
        "col1": "Nitrogen_kg_ha",
        "col2": "Phosphorus_kg_ha"
    }
    r = requests.post(f"{API_URL}/analyze/nonparametric", headers=headers, files=files_nonparam, data=nonparam_data)
    print(f"Mann-Whitney response status: {r.status_code}")
    assert r.status_code == 200
    nonparam_res = r.json()
    assert "normality" in nonparam_res
    assert "statistics" in nonparam_res
    assert "plot" in nonparam_res
    assert "iqr_group1" in nonparam_res["statistics"]
    assert "effect_size" in nonparam_res["statistics"]
    print("Mann-Whitney U test executed successfully!")
    print("Statistics output:", nonparam_res["statistics"])

    # Save plot
    nonparam_plot_bytes = base64.b64decode(nonparam_res["plot"])
    with open("test_nonparametric_plot.png", "wb") as f:
        f.write(nonparam_plot_bytes)

    # 9. Test /analyze/nonparametric (Wilcoxon Signed-Rank Test)
    csv_buf.seek(0)
    files_wilcoxon = {"file": ("dataset_soil.csv", csv_buf.getvalue(), "text/csv")}
    wilcoxon_data = {
        "test_type": "wilcoxon",
        "col1": "Nitrogen_kg_ha",
        "col2": "Phosphorus_kg_ha"
    }
    r = requests.post(f"{API_URL}/analyze/nonparametric", headers=headers, files=files_wilcoxon, data=wilcoxon_data)
    print(f"Wilcoxon response status: {r.status_code}")
    assert r.status_code == 200
    wilcoxon_res = r.json()
    assert "normality" in wilcoxon_res
    assert "differences" in wilcoxon_res["normality"]
    assert "statistics" in wilcoxon_res
    assert "median_difference" in wilcoxon_res["statistics"]
    assert "effect_size" in wilcoxon_res["statistics"]
    print("Wilcoxon test executed successfully!")

    # 10. Test /analyze/nonparametric (Kruskal-Wallis Smart Routing with 3+ cols)
    csv_buf.seek(0)
    files_kruskal = {"file": ("dataset_soil.csv", csv_buf.getvalue(), "text/csv")}
    kruskal_data = {
        "test_type": "mann_whitney", # Smart routed to kruskal
        "cols_str": "Nitrogen_kg_ha,Phosphorus_kg_ha,CropYield_t_ha"
    }
    r = requests.post(f"{API_URL}/analyze/nonparametric", headers=headers, files=files_kruskal, data=kruskal_data)
    print(f"Kruskal-Wallis response status: {r.status_code}")
    assert r.status_code == 200
    kruskal_res = r.json()
    assert kruskal_res["statistics"]["test_name"] == "Kruskal-Wallis H-Test"
    assert "medians" in kruskal_res["statistics"]
    assert "effect_size" in kruskal_res["statistics"]
    print("Kruskal-Wallis smart routed test executed successfully!")

    # 11. Test /analyze/nonparametric (Friedman repeated measures)
    csv_buf.seek(0)
    files_friedman = {"file": ("dataset_soil.csv", csv_buf.getvalue(), "text/csv")}
    friedman_data = {
        "test_type": "friedman",
        "cols_str": "Nitrogen_kg_ha,Phosphorus_kg_ha,CropYield_t_ha"
    }
    r = requests.post(f"{API_URL}/analyze/nonparametric", headers=headers, files=files_friedman, data=friedman_data)
    print(f"Friedman response status: {r.status_code}")
    assert r.status_code == 200
    friedman_res = r.json()
    assert friedman_res["statistics"]["test_name"] == "Friedman Test (Repeated Measures)"
    assert "effect_size" in friedman_res["statistics"]
    print("Friedman test executed successfully!")

    # 12. Test /analyze/nonparametric (Chi-Square)
    csv_buf.seek(0)
    files_chisq = {"file": ("dataset_soil.csv", csv_buf.getvalue(), "text/csv")}
    chisq_data = {
        "test_type": "chi_square",
        "cols_str": "FertilizerType,YieldStatus"
    }
    r = requests.post(f"{API_URL}/analyze/nonparametric", headers=headers, files=files_chisq, data=chisq_data)
    print(f"Chi-Square response status: {r.status_code}")
    assert r.status_code == 200
    chisq_res = r.json()
    assert chisq_res["statistics"]["test_name"] == "Chi-Square Test of Independence"
    assert "contingency_table" in chisq_res["statistics"]
    assert "effect_size" in chisq_res["statistics"]
    print("Chi-Square test executed successfully!")

    print("--- Integration Verification Completed Successfully ---")

if __name__ == "__main__":
    test_integration()
