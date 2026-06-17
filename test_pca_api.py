import requests
import pandas as pd
import io

def test_pca_and_duncan():
    print("--- Starting PCA and Duncan API Verification ---")
    
    # 1. Login
    login_url = "http://127.0.0.1:8000/auth/login"
    login_data = {
        "username": "ravi.scholar@icar-iiss.res.in",
        "password": "PhDpassword2026!"
    }
    
    response = requests.post(login_url, data=login_data)
    if response.status_code != 200:
        print("Login failed! Make sure the backend server is running and the user is registered.")
        return
        
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("Logged in successfully.")
    
    # 2. Test PCA endpoint with test1.csv (numeric data)
    pca_url = "http://127.0.0.1:8000/analyze/pca"
    files = {
        "file": ("test1.csv", open("../test1.csv", "rb"), "text/csv")
    }
    data = {
        "columns_str": "Mn PPM,Zn PPM,Cd ppm,Cr ppm",
        "scale": "true"
    }
    
    print("\nTesting PCA Analysis...")
    pca_res = requests.post(pca_url, headers=headers, files=files, data=data)
    print(f"Status: {pca_res.status_code}")
    if pca_res.status_code == 200:
        pca_data = pca_res.json()
        print("PCA Success!")
        print(f"Eigenvalues: {pca_data['eigenvalues']}")
        print(f"Explained Variance Ratio: {pca_data['explained_variance_ratio']}")
        print(f"Cumulative Variance Ratio: {pca_data['cumulative_variance_ratio']}")
        print(f"PCs returned: {pca_data['pc_names']}")
        print(f"Variables returned: {pca_data['variable_names']}")
        print(f"Sample scores count: {len(pca_data['sample_scores'])}")
        print("Plot image returned successfully: ", "plot" in pca_data)
    else:
        print(f"PCA Error: {pca_res.text}")
        
    # 3. Test ANOVA with Duncan posthoc (requires generating Block column and filtering)
    df = pd.read_csv("../test_anova.csv")
    # Add Block column dynamically
    df['Block'] = 'Block' + (df.groupby(['Fertilizer', 'Water']).cumcount() + 1).astype(str)
    # Filter for One-Factor RBD
    df_rbd = df[df['Water'] == 'Rainfed']
    
    csv_buf = io.StringIO()
    df_rbd.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    
    anova_url = "http://127.0.0.1:8000/analyze/anova"
    anova_files = {
        "file": ("test_anova_modified.csv", csv_buf.getvalue(), "text/csv")
    }
    anova_data = {
        "test_type": "rbd_oneway",
        "dep_var": "Yield",
        "ind_var1": "Fertilizer",
        "rep_var": "Block",
        "posthoc_method": "duncan"
    }
    
    print("\nTesting ANOVA with Duncan posthoc...")
    anova_res = requests.post(anova_url, headers=headers, files=anova_files, data=anova_data)
    print(f"Status: {anova_res.status_code}")
    if anova_res.status_code == 200:
        anova_out = anova_res.json()
        print("ANOVA with Duncan Success!")
        print("Duncan Posthoc Letters:")
        for tr, letter in anova_out["posthoc_letters"].items():
            print(f"  {tr}: {letter}")
    else:
        print(f"ANOVA Error: {anova_res.text}")

if __name__ == "__main__":
    test_pca_and_duncan()
