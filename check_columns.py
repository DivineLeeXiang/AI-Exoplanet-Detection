import pandas as pd

df = pd.read_csv(
    r"detection/training/cumulative_2020.12.30_14.14.11.csv",
    comment="#"
)

print(df.columns.tolist())