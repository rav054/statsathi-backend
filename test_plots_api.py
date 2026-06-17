import sys
import requests
import pandas as pd
import io
import base64

API_URL = "http://127.0.0.1:8000"

def run_tests():
    print("--- Starting Plots API Verification ---")

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

    # 2. Generate a mock dataset
    data = {
        "Fertilizer": ["Control", "Control", "Low_N", "Low_N", "High_N", "High_N", "Control", "Control", "Low_N", "Low_N", "High_N", "High_N"],
        "Water": ["Rainfed", "Irrigated", "Rainfed", "Irrigated", "Rainfed", "Irrigated", "Rainfed", "Irrigated", "Rainfed", "Irrigated", "Rainfed", "Irrigated"],
        "Yield": [45.2, 52.3, 62.8, 74.5, 78.9, 94.6, 43.8, 50.8, 61.4, 73.1, 77.2, 92.5],
        "Days_to_Flower": [60, 58, 55, 54, 52, 50, 61, 59, 56, 55, 53, 51]
    }
    df = pd.DataFrame(data)
    
    def send_plot_request(payload_data):
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        files = {"file": ("test_plots.csv", csv_buf.getvalue(), "text/csv")}
        r = requests.post(f"{API_URL}/analyze/plot", headers=headers, files=files, data=payload_data)
        return r

    # Test 1: Single Boxplot with Emerald palette and custom text color
    print("\nTesting Single Boxplot with custom text color...")
    payload = {
        "plot_type": "boxplot",
        "x_var": "Yield",
        "palette": "emerald",
        "text_color": "#312E81",
        "title": "Yield Distribution (Boxplot)",
        "ylabel": "Crop Yield (t/ha)",
        "show_grid": "True"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Single Boxplot Success!")

    # Test 2: Grouped Boxplot with Spring palette and Custom aspect ratio (wide)
    print("\nTesting Grouped Boxplot with Spring palette...")
    payload = {
        "plot_type": "boxplot",
        "x_var": "Fertilizer",
        "y_var": "Yield",
        "hue_var": "Water",
        "palette": "spring",
        "aspect_ratio": "wide",
        "title": "Grouped Yield by Treatment",
        "xlabel": "Fertilizer Treatment",
        "ylabel": "Yield",
        "legend_loc": "upper left"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Grouped Boxplot Success!")

    # Test 3: Histogram with bins=12, KDE=True, color Crimson, and Square layout
    print("\nTesting Histogram with KDE...")
    payload = {
        "plot_type": "histogram",
        "x_var": "Yield",
        "bins": 12,
        "kde": "True",
        "palette": "crimson",
        "aspect_ratio": "square",
        "title": "Yield Histogram",
        "show_grid": "True"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Histogram Success!")

    # Test 4: Q-Q Plot
    print("\nTesting Q-Q Plot...")
    payload = {
        "plot_type": "qqplot",
        "x_var": "Yield",
        "palette": "teal",
        "title": "Normal Q-Q Plot of Yield"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Q-Q Plot Success!")

    # Test 5: Scatter Plot with regression trend line, grouping, Teal color
    print("\nTesting Scatter Plot...")
    payload = {
        "plot_type": "scatter",
        "x_var": "Days_to_Flower",
        "y_var": "Yield",
        "hue_var": "Fertilizer",
        "fit_reg": "True",
        "palette": "teal",
        "title": "Yield vs Days to Flower",
        "xlabel": "Flowering Time (days)",
        "ylabel": "Yield (t/ha)"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Scatter Plot Success!")

    # Test 6: Pie Chart
    print("\nTesting Pie Chart...")
    payload = {
        "plot_type": "pie",
        "x_var": "Fertilizer",
        "palette": "sunset",
        "title": "Fertilizer Treatments Breakdown"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Pie Chart Success!")

    # Test 7: Line Graph with grouping
    print("\nTesting Line Graph...")
    payload = {
        "plot_type": "line",
        "x_var": "Days_to_Flower",
        "y_var": "Yield",
        "hue_var": "Water",
        "palette": "charcoal",
        "title": "Yield Trends over Flowering Days"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Line Graph Success!")

    # Test 8: Invalid Requests (Expected to fail with 400)
    print("\nTesting Invalid Request (non-numeric for Scatter)...")
    payload = {
        "plot_type": "scatter",
        "x_var": "Fertilizer", # Non-numeric
        "y_var": "Yield"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 400
    print("Invalid Request Rejected Successfully!")

    # Test 9: Bar chart with Standard Error errorbars
    print("\nTesting Bar Chart with Standard Error error bars...")
    payload = {
        "plot_type": "barplot",
        "x_var": "Fertilizer",
        "y_var": "Yield",
        "hue_var": "Water",
        "errorbar_toggle": "True",
        "errorbar_type": "se",
        "palette": "indigo",
        "title": "Yield by Fertilizer & Water (Bar Chart)"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Bar Chart Success!")

    # Test 10: Violin plot
    print("\nTesting Violin Plot...")
    payload = {
        "plot_type": "violin",
        "x_var": "Fertilizer",
        "y_var": "Yield",
        "hue_var": "Water",
        "palette": "coolwarm",
        "title": "Yield Distribution Violin Plot"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Violin Plot Success!")

    # Test 11: Multi-line plot
    print("\nTesting Multi-line Plot...")
    payload = {
        "plot_type": "multiline",
        "x_var": "Days_to_Flower",
        "y_vars_str": "Yield,Yield",
        "palette": "sunset",
        "title": "Multi-Line High-Density Plot"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("Multi-line Plot Success!")

    # Test 12: PCA Biplot
    print("\nTesting PCA Biplot...")
    payload = {
        "plot_type": "pcabiplot",
        "x_var": "Yield",
        "y_vars_str": "Yield,Days_to_Flower",
        "hue_var": "Fertilizer",
        "palette": "viridis",
        "title": "PCA Biplot Projection"
    }
    res = send_plot_request(payload)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200
    assert "plot" in res.json()
    print("PCA Biplot Success!")

    # Test 13: Download SVG at 300 DPI
    print("\nTesting Download SVG at 300 DPI...")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    files = {"file": ("test_plots.csv", csv_buf.getvalue(), "text/csv")}
    download_payload = {
        "plot_type": "scatter",
        "x_var": "Days_to_Flower",
        "y_var": "Yield",
        "download_format": "svg",
        "dpi": 300
    }
    r = requests.post(f"{API_URL}/analyze/plot/download", headers=headers, files=files, data=download_payload)
    print(f"Status: {r.status_code}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/svg+xml"
    assert len(r.content) > 0
    print("Download SVG Success!")

    # Test 14: Download PDF at 600 DPI
    print("\nTesting Download PDF at 600 DPI...")
    files = {"file": ("test_plots.csv", csv_buf.getvalue(), "text/csv")}
    download_payload = {
        "plot_type": "barplot",
        "x_var": "Fertilizer",
        "y_var": "Yield",
        "download_format": "pdf",
        "dpi": 600,
        "errorbar_toggle": "True",
        "errorbar_type": "sd"
    }
    r = requests.post(f"{API_URL}/analyze/plot/download", headers=headers, files=files, data=download_payload)
    print(f"Status: {r.status_code}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert len(r.content) > 0
    print("Download PDF Success!")

    print("\n--- All Plots API Verification Tests Passed! ---")

if __name__ == "__main__":
    run_tests()

