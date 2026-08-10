import sys
import io
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_all():
    print("==================================================")
    print("TESTING SEM API ENDPOINTS VIA FASTAPI TESTCLIENT")
    print("==================================================")

    # 1. Test GET /api/sem/templates
    print("\n--- 1. Testing GET /api/sem/templates ---")
    res = client.get("/api/sem/templates")
    print(f"Status Code: {res.status_code}")
    assert res.status_code == 200
    templates = res.json()
    print("Templates returned:")
    for k, v in templates.items():
        print(f"  [{k}]: {v}")
    
    assert "simple_regression" in templates
    assert "mediation" in templates
    assert "latent_factor" in templates
    assert "full_sem" in templates
    print("SUCCESS: GET /api/sem/templates PASSED!")

    # 2. Generate sample CSV data
    np.random.seed(42)
    n = 120
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(5, 1, n)
    x3 = np.random.normal(2, 0.5, n)
    y = 0.5 * x1 + 0.8 * x2 - 0.3 * x3 + np.random.normal(0, 1, n)

    df = pd.DataFrame({'Y': y, 'X1': x1, 'X2': x2, 'X3': x3})
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    # 3. Test POST /api/sem/fit
    print("\n--- 2. Testing POST /api/sem/fit ---")
    files = {
        'file': ('data.csv', io.BytesIO(csv_bytes), 'text/csv')
    }
    data = {
        'model': 'Y ~ X1 + X2 + X3'
    }

    res = client.post("/api/sem/fit", files=files, data=data)
    print(f"Status Code: {res.status_code}")
    assert res.status_code == 200
    fit_json = res.json()
    
    print("\nFit Result JSON:")
    import json
    print(json.dumps(fit_json, indent=2))

    assert fit_json["success"] is True
    assert fit_json["n_obs"] == 120
    assert "fit_indices" in fit_json
    assert "parameters" in fit_json
    assert len(fit_json["parameters"]) > 0

    # Verify parameter keys
    p0 = fit_json["parameters"][0]
    for key in ["lval", "rval", "Estimate", "Std.Err", "z-value", "p-value", "significant"]:
        assert key in p0, f"Missing key '{key}' in parameter output"

    print("SUCCESS: POST /api/sem/fit PASSED!")

    # 4. Test POST /api/sem/diagram
    print("\n--- 3. Testing POST /api/sem/diagram ---")
    files_diag = {
        'file': ('data.csv', io.BytesIO(csv_bytes), 'text/csv')
    }
    data_diag = {
        'model': 'Y ~ X1 + X2 + X3'
    }

    res_diag = client.post("/api/sem/diagram", files=files_diag, data=data_diag)
    print(f"Status Code: {res_diag.status_code}")
    assert res_diag.status_code == 200
    diag_json = res_diag.json()
    print("Diagram Endpoint JSON Response:")
    print(json.dumps(diag_json, indent=2))

    print("\nALL SEM API TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_all()
