import sys
import json
import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.auth import get_current_user
from app.models import User

client = TestClient(app)

# Mock authenticated user
mock_user = User(
    id="test-user-id",
    email="ravi.scholar@icar-iiss.res.in",
    full_name="Dr. Ravi Kumar",
    institution="ICAR-IISS Bhopal",
    hashed_password="hashed_pass_dummy"
)

app.dependency_overrides[get_current_user] = lambda: mock_user

def test_transformations_unit():
    print("=== Starting FastAPI TestClient Data Transformation Unit Tests ===")

    # 1. Prepare positive dataset
    df_pos = pd.DataFrame({
        "Yield": [10.5, 25.0, 42.1, 18.3, 33.7, 50.2],
        "Height": [120, 145, 160, 130, 155, 175],
        "Proportion": [0.15, 0.25, 0.40, 0.65, 0.80, 0.95]
    })
    csv_pos_buf = io.BytesIO()
    df_pos.to_csv(csv_pos_buf, index=False)
    csv_pos_bytes = csv_pos_buf.getvalue()

    # 2. Prepare dataset with negative values
    df_neg = pd.DataFrame({
        "Temperature": [-5.0, 0.0, 12.5, 22.0, -1.2, 18.4]
    })
    csv_neg_buf = io.BytesIO()
    df_neg.to_csv(csv_neg_buf, index=False)
    csv_neg_bytes = csv_neg_buf.getvalue()

    # Test all 8 methods
    methods_to_test = ['log10', 'ln', 'sqrt', 'arcsine', 'boxcox', 'yeojohnson', 'zscore', 'minmax']
    for m in methods_to_test:
        cols_arg = ["Yield"] if m != 'arcsine' else ["Proportion"]
        files = {"file": ("test_pos.csv", csv_pos_bytes, "text/csv")}
        data = {
            "columns": json.dumps(cols_arg),
            "method": m
        }
        res = client.post("/analyze/transform", files=files, data=data)
        assert res.status_code == 200, f"Method '{m}' failed: {res.text}"
        res_json = res.json()
        assert len(res_json["transformed_columns"]) == 1
        new_col = res_json["transformed_columns"][0]
        print(f"[OK] Method '{m}' succeeded! Created transformed column: '{new_col}'")

    # Test error handling: Log10 on negative values
    files_neg = {"file": ("test_neg.csv", csv_neg_bytes, "text/csv")}
    data_err = {
        "columns": json.dumps(["Temperature"]),
        "method": "log10"
    }
    res_err = client.post("/analyze/transform", files=files_neg, data=data_err)
    assert res_err.status_code == 400
    detail = res_err.json().get("detail", "")
    assert "Cannot apply Log/Box-Cox to zero or negative values" in detail or "Try Yeo-Johnson instead" in detail
    print(f"[OK] Error handling verified! 400 Detail: '{detail}'")

    # Test Yeo-Johnson on negative values (should succeed)
    data_yj = {
        "columns": json.dumps(["Temperature"]),
        "method": "yeojohnson"
    }
    res_yj = client.post("/analyze/transform", files=files_neg, data=data_yj)
    assert res_yj.status_code == 200
    print(f"[OK] Yeo-Johnson on negative values succeeded! Created column: '{res_yj.json()['transformed_columns'][0]}'")

    print("\n=== ALL FASTAPI TRANSFORMATION UNIT TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_transformations_unit()
