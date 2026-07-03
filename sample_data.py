import pandas as pd

X_test = pd.read_csv("data/processed/X_test.csv")
Y_test = pd.read_csv("data/processed/Y_test.csv").squeeze()

sample_size = 5000

sample_index = X_test.sample(
    n=sample_size,
    random_state=42
).index

X_sample = X_test.loc[sample_index]
Y_sample = Y_test.loc[sample_index]

X_sample.to_csv("data/app_data/X_test_sample.csv", index=False)
Y_sample.to_csv("data/app_data/Y_test_sample.csv", index=False)

print("Sample dataset created successfully!")