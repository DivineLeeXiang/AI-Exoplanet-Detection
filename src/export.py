import os
import pandas as pd


def export_results(results):

    os.makedirs("outputs", exist_ok=True)

    csv_path = os.path.join("outputs", "exoplanet_candidates.csv")

    df = pd.DataFrame(results)

    df.to_csv(csv_path, index=False)

    print(f"\nCSV exported successfully!")
    print(f"Location: {csv_path}")

    return csv_path