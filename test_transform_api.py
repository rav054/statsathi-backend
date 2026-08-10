import sys
import requests
import json
import io
import pandas as pd

API_URL = "http://127.0.0.1:8000"

def test_transformations():
    print("=== Starting Data Transformation & Scaling API Verification ===")

    # 1. Health check
    try:
        r = requests.get(f"{API_URL}/")
        assert r.status_code == 200
        print("[✓] Backend is online!")
    except Exception as e:
        print(f"[X] Backend connection failed: {e}")
        sys.exit(1)

    # 2. Login
    login_data = {
        "username": "ravi.scholar@icar-iiss.res.in",
        "password": "PhDpassword2026!"
    }
    r = requests.post(f"{API_URL}/auth/login", data=login_data)
    if r.status_code != 200:
        # Try signing up
        signup_payload = {
            "email": "ravi.scholar@icar-iiss.res.in",
            "full_name": "Dr. Ravi Kumar",
            "institution": "ICAR-IISS Bhopal",
            "password": "PhDpassword2026!"
        }
        requests.post(f"{API_URL}/auth/signup", json=signup_payload)
        r = requests.post(f"{API_URL}/auth/login", data=login_data)
    
    assert r.status_code == 200
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[✓] Auth token obtained!")

    # 3. Prepare test datasets
    # Dataset A: Positive values (suited for Log10, Ln, Sqrt, BoxCox, YeoJohnson, ZScore, MinMax)
    df_pos = pd.DataFrame({
        "Yield": [10.5, 25.0, 42.1, 18.3, 33.7, 50.2],
        "Height": [120, 145, 160, 130, 155, 175],
        "Proportion": [0.15, 0.25, 0.40, 0.65, 0.80, 0.95]
    })
    csv_pos_buf = io.BytesIO()
    df_pos.to_csv(csv_pos_buf, index=False)
    csv_pos_bytes = csv_pos_buf.getvalue()

    # Dataset B: Contains Zero & Negative values (for testing error handling & Yeo-Johnson)
    df_neg = pd.DataFrame({
        "Temperature": [-5.0, 0.0, 12.5, 22.0, -1.2, 18.4]
    })
    csv_neg_buf = io.BytesIO()
    df_neg.to_csv(csv_neg_buf, index=False)
    csv_neg_bytes = csv_neg_buf.getvalue()

    # Test Methods on Positive Dataset
    methods_to_test = ['log10', 'ln', 'sqrt', 'arcsine', 'boxcox', 'yeojohnson', 'zscore', 'minmax', 'snv', 'msc', 'sg_smooth', 'sg_1der', 'sg_2der']
    for m in methods_to_test:
        cols_arg = ["Yield"] if m != 'arcsine' else ["Proportion"]
        files = {"file": ("test_pos.csv", csv_pos_bytes, "text/csv")}
        data = {
            "columns": json.dumps(cols_arg),
            "method": m
        }
        res = requests.post(f"{API_URL}/analyze/transform", headers=headers, files=files, data=data)
        assert res.status_code == 200, f"Failed on method {m}: {res.text}"
        res_json = res.json()
        assert len(res_json["transformed_columns"]) == 1
        new_col = res_json["transformed_columns"][0]
        print(f"[✓] Method '{m}' succeeded! Transformed column created: '{new_col}'")

    # Test Error Handling: Log10 on negative numbers
    files = {"file": ("test_neg.csv", csv_neg_bytes, "text/csv")}
    data = {
        "columns": json.dumps(["Temperature"]),
        "method": "log10"
    }
    res_err = requests.post(f"{API_URL}/analyze/transform", headers=headers, files=files, data=data)
    assert res_err.status_code == 400
    err_detail = res_err.json().get("detail", "")
    assert "Cannot apply Log/Box-Cox to zero or negative values" in err_detail or "Try Yeo-Johnson instead" in err_detail
    print(f"[✓] Error handling verified! Clean 400 detail returned: '{err_detail}'")

    # Test Yeo-Johnson on negative numbers (should succeed!)
    data_yj = {
        "columns": json.dumps(["Temperature"]),
        "method": "yeojohnson"
    }
    res_yj = requests.post(f"{API_URL}/analyze/transform", headers=headers, files=files, data=data_yj)
    assert res_yj.status_code == 200
    print(f"[✓] Yeo-Johnson successfully transformed negative values! Transformed column: '{res_yj.json()['transformed_columns'][0]}'")

    print("\n=== All Data Transformation & Scaling API Tests PASSED Successfully! ===")

if __name__ == "__main__":
    test_transformations()
