import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load CSV
df = pd.read_csv("cleaned_eamcet_data.csv")

# Rename columns for consistency
df = df.rename(columns={
    "Branch Name": "Branch",
    "Category": "Caste",
    "Closing Rank": "Closing_Rank"
})

# Drop rows where Closing_Rank is not a number
df = df[pd.to_numeric(df["Closing_Rank"], errors='coerce').notnull()]
df["Closing_Rank"] = df["Closing_Rank"].astype(float)

# Encode categorical variables
df_encoded = pd.get_dummies(df[["Caste", "Gender", "Branch"]])

# Target variable
y = df["Closing_Rank"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and feature order
joblib.dump((model, df_encoded.columns.tolist()), "college_predictor_model.pkl")

print("✅ Model trained and saved successfully.")
