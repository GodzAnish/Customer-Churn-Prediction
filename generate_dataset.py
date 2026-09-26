"""
Generate Sample Customer Churn Dataset
Creates a realistic customer dataset for churn prediction analysis
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 5000

# Generate customer data
data = {
    'CustomerID': range(1, n_samples + 1),
    'Age': np.random.randint(18, 70, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Tenure': np.random.randint(1, 72, n_samples),  # months
    'MonthlyCharges': np.round(np.random.uniform(20, 150, n_samples), 2),
    'TotalCharges': np.round(np.random.uniform(100, 8000, n_samples), 2),
    'ContractType': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.20]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
}

# Create DataFrame
df = pd.DataFrame(data)

# Create churn target variable (with realistic correlations)
churn_probability = (
    (df['ContractType'] == 'Month-to-month').astype(int) * 0.3 +
    (df['Tenure'] < 12).astype(int) * 0.25 +
    (df['MonthlyCharges'] > 100).astype(int) * 0.15 +
    (df['TechSupport'] == 'No').astype(int) * 0.1 +
    (df['OnlineSecurity'] == 'No').astype(int) * 0.1 +
    np.random.uniform(0, 0.1, n_samples)  # random noise
)

# Normalize to 0-1 range
churn_probability = (churn_probability - churn_probability.min()) / (churn_probability.max() - churn_probability.min())

# Generate churn labels
df['Churn'] = (churn_probability > 0.6).astype(int)

# Save to CSV
df.to_csv('data/customer_churn_data.csv', index=False)

print(f"✅ Generated {n_samples} customer records")
print(f"📊 Churn Rate: {df['Churn'].mean()*100:.2f}%")
print(f"💾 Saved to: data/customer_churn_data.csv")
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nChurn Distribution:")
print(df['Churn'].value_counts())
