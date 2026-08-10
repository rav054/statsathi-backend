import os
import io
import pandas as pd
import numpy as np

def run_test():
    print("=== Testing semopy engine directly ===")
    try:
        import semopy
    except ImportError:
        print("ERROR: semopy not installed yet.")
        return

    # Generate synthetic dataset for testing simple regression & mediation
    np.random.seed(42)
    n = 120
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(5, 1, n)
    x3 = np.random.normal(2, 0.5, n)
    y = 0.5 * x1 + 0.8 * x2 - 0.3 * x3 + np.random.normal(0, 1, n)

    df = pd.DataFrame({'Y': y, 'X1': x1, 'X2': x2, 'X3': x3})
    df.to_csv('test_sem_data.csv', index=False)
    print("Created test_sem_data.csv with 120 rows.")

    desc = "Y ~ X1 + X2 + X3"
    mod = semopy.Model(desc)
    res = mod.fit(df)
    print(f"Fit result status: {res}")

    params = mod.inspect()
    print("\n--- Model Parameters ---")
    print(params)

    stats = semopy.calc_stats(mod)
    print("\n--- Fit Statistics ---")
    print(stats)

    try:
        g = semopy.semplot(mod, "temp.gv")
        print(f"\n--- Diagram Object ---: {type(g)}")
        if hasattr(g, 'pipe'):
            svg = g.pipe(format='svg').decode('utf-8')
            print(f"SVG generated successfully! Length: {len(svg)} chars")
    except Exception as e:
        print(f"Diagram generation note: {e}")

if __name__ == "__main__":
    run_test()
