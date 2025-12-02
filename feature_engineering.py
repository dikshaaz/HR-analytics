# FeatureEngineering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features_and_target(df):
    """
    Prepares features (X) and target (y) for modeling:
    - Converts target 'Attrition' from Yes/No to 1/0
    - One-hot encodes categorical variables
    - Scales numeric features
    """
    # Target column
    target_col = 'Attrition'

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Convert Yes/No to 1/0
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

    # Identify categorical columns excluding the target
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Identify numeric columns excluding target
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols.remove(target_col)

    # Scale numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Split X and y
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print("✔ Feature engineering completed!")
    print(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")

    return X, y