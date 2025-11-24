# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

input_path = "/opt/ml/processing/input/diabetes.csv"
output_train = "/opt/ml/processing/output/train/train.csv"
output_test = "/opt/ml/processing/output/test/test.csv"
output_val = "/opt/ml/processing/output/validation/validation.csv"

os.makedirs(os.path.dirname(output_train), exist_ok=True)
os.makedirs(os.path.dirname(output_test), exist_ok=True)
os.makedirs(os.path.dirname(output_val), exist_ok=True)

# Load dataset
df = pd.read_csv(input_path)
df.drop(['SkinThickness','Insulin'], axis=1, inplace=True)
# Replace 0s in selected columns with median values
zero_as_missing = ["Glucose", "BloodPressure", "BMI"]
for col in zero_as_missing:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Split train/test/val (70/20/10)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Outcome"])
test_df, val_df = train_test_split(temp_df, test_size=0.33, random_state=42, stratify=temp_df["Outcome"])

train_df.to_csv(output_train, index=False)
test_df.to_csv(output_test, index=False)
val_df.to_csv(output_val, index=False)

print("✅ Preprocessing done and splits saved!")