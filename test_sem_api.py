import sys
import requests
import json

API_URL = "http://127.0.0.1:8000"

def test_sem():
    print("--- Starting SEM API Verification ---")

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

    # 3. Test SEM Analysis
    sem_url = f"{API_URL}/analyze/sem"
    
    spec = {
        "latent_variables": [
            {"id": "L1", "name": "Heavy_Metals", "indicators": ["Cd ppm", "Cr ppm", "Pb ppm"]},
            {"id": "L2", "name": "Essential_Nutrients", "indicators": ["Mn PPM", "Zn PPM"]}
        ],
        "paths": [
            {"from": "L1", "to": "L2"}
        ]
    }
    
    files = {
        "file": ("test1.csv", open("../test1.csv", "rb"), "text/csv")
    }
    data = {
        "sem_type": "pls",
        "specification": json.dumps(spec)
    }

    print("\nTesting SEM (PLS) Path Modeling...")
    r = requests.post(sem_url, headers=headers, files=files, data=data)
    print(f"SEM Response Status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        print("SEM analysis succeeded!")
        
        # Verify engine and output structure
        print(f"  Engine run: {res.get('engine')}")
        assert "path_coefficients" in res
        assert "outer_loadings" in res
        assert "reliability_indices" in res
        assert "fit_indices" in res
        
        paths_out = res["path_coefficients"]
        print(f"  Path coefficient (L1 -> L2): {paths_out[0]['coefficient']:.4f}")
        print(f"  Path p-value (L1 -> L2): {paths_out[0]['p_value']}")
        
        rel_indices = res["reliability_indices"]
        for rel in rel_indices:
            print(f"  Latent {rel['latent_name']}: Alpha={rel['cronbach_alpha']:.4f}, CR={rel['composite_reliability']:.4f}, AVE={rel['ave']:.4f}")
            
        fit = res["fit_indices"]
        print(f"  Fit Indices: Chi2={fit['chi_square']:.2f}, CFI={fit['cfi']:.3f}, TLI={fit['tli']:.3f}, RMSEA={fit['rmsea']:.3f}")
    else:
        print(f"SEM Error: {r.text}")
        sys.exit(1)

    print("\nAll SEM API Verification Tests Passed!")

if __name__ == "__main__":
    test_sem()
