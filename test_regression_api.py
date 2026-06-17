import sys
import requests
import json

API_URL = "http://127.0.0.1:8000"

def test_regression():
    print("--- Starting Regression API Verification ---")

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

    # 3. Test Simple Regression
    regression_url = f"{API_URL}/analyze/regression"
    files = {
        "file": ("test1.csv", open("../test1.csv", "rb"), "text/csv")
    }
    data = {
        "regression_type": "simple",
        "dep_vars_str": "Mn PPM",
        "ind_vars_str": "Zn PPM"
    }

    print("\nTesting Simple Linear Regression...")
    r = requests.post(regression_url, headers=headers, files=files, data=data)
    print(f"Response Status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        print("Simple Regression Succeeded!")
        assert res["regression_type"] == "simple"
        assert "r2" in res
        assert "coefficients" in res
        print(f"  R2: {res['r2']:.4f}")
        print(f"  Intercept: {res['coefficients'][0]['coefficient']:.4f}")
        print(f"  Slope (Zn PPM): {res['coefficients'][1]['coefficient']:.4f}")
        print(f"  Slope p-value: {res['coefficients'][1]['p_value']}")
    else:
        print(f"Error: {r.text}")
        sys.exit(1)

    # 4. Test Multiple Regression
    files = {
        "file": ("test1.csv", open("../test1.csv", "rb"), "text/csv")
    }
    data = {
        "regression_type": "multiple",
        "dep_vars_str": "Mn PPM",
        "ind_vars_str": "Zn PPM,Cd ppm,Cr ppm"
    }

    print("\nTesting Multiple Linear Regression...")
    r = requests.post(regression_url, headers=headers, files=files, data=data)
    print(f"Response Status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        print("Multiple Regression Succeeded!")
        assert res["regression_type"] == "multiple"
        assert "r2" in res
        print(f"  R2: {res['r2']:.4f}")
        print(f"  Adjusted R2: {res['adj_r2']:.4f}")
        print(f"  F-Statistic: {res['f_statistic']:.3f} (p: {res['f_pvalue']})")
        print(f"  Coefficients count: {len(res['coefficients'])}")
    else:
        print(f"Error: {r.text}")
        sys.exit(1)

    # 5. Test PLSR
    files = {
        "file": ("test1.csv", open("../test1.csv", "rb"), "text/csv")
    }
    data = {
        "regression_type": "plsr",
        "dep_vars_str": "Mn PPM,Zn PPM",
        "ind_vars_str": "Cd ppm,Cr ppm,Pb ppm",
        "n_components": 2
    }

    print("\nTesting PLS Regression...")
    r = requests.post(regression_url, headers=headers, files=files, data=data)
    print(f"Response Status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        print("PLSR Succeeded!")
        assert res["regression_type"] == "plsr"
        assert "r2_values" in res
        print(f"  R2 for targets: {res['r2_values']}")
        print(f"  Explained variance (X): {res['explained_variance_x']}")
        print(f"  Number of target coefficients: {len(res['coefficients'])}")
    else:
        print(f"Error: {r.text}")
        sys.exit(1)

    print("\nAll Regression API Verification Tests Passed!")

if __name__ == "__main__":
    test_regression()
