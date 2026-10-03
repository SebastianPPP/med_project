import pandas as pd

# data
df = pd.read_csv('SEER Breast Cancer Dataset .csv')

print("=== BASIC INFORMATION AND DATASET SIZE ===")
print(f"Rows (patients): {df.shape[0]}")
print(f"Columns (variables): {df.shape[1]}")
print("==========================================")

# NaN values / faulty data
print("=== NaN VALUES / FAULTY DATA ===")
print(df.isnull().sum())
print("==========================================")

# drop 4th column (SEER cause-specific death classification)
df.drop(df.columns[3], axis=1, inplace=True)

print("Dropped 4th column (faulty values)")
print("===========================================")

# save to csv
df.to_csv('SEER Breast Cancer Dataset Cleaned.csv', index=False)
print("Saved cleaned dataset to 'SEER Breast Cancer Dataset Cleaned.csv'")
print("===========================================")