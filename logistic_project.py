import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("Telco_customer_churn.csv")
df.drop(columns=['Churn Reason', 'Lat Long', 'CustomerID', 'Zip Code', 'Count', 'Churn Label'], inplace=True)

target_col = "Churn Value"

df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

constant_cols = [col for col in df.columns if df[col].nunique() == 1]
df.drop(columns=constant_cols, inplace=True)
print("Dropped constant columns:", constant_cols)

high_card_drop = [col for col in ['City'] if col in df.columns]
df.drop(columns=high_card_drop, inplace=True)
print("Dropped high-cardinality columns:", high_card_drop)

gender_code = {'Male': 0, 'Female': 1}
yes_no_code = {'No': 0, 'Yes': 1}
df['Gender'] = df['Gender'].map(gender_code)

binary_cols = [
    col for col in df.select_dtypes(include="object").columns
    if df[col].nunique() == 2
]
for col in binary_cols:
    df[col] = df[col].map(yes_no_code)

categorical_cols = [
    col for col in df.select_dtypes(include='object')
    if df[col].nunique() >= 3
]

train_val, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val, test_size=0.25, random_state=42)

median_charge = train_df['Total Charges'].median()
train_df['Total Charges'] = train_df['Total Charges'].fillna(median_charge)
val_df['Total Charges'] = val_df['Total Charges'].fillna(median_charge)
test_df['Total Charges'] = test_df['Total Charges'].fillna(median_charge)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(train_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_encoded = pd.DataFrame(encoder.transform(train_df[categorical_cols]), columns=encoded_cols, index=train_df.index)
val_encoded = pd.DataFrame(encoder.transform(val_df[categorical_cols]), columns=encoded_cols, index=val_df.index)
test_encoded = pd.DataFrame(encoder.transform(test_df[categorical_cols]), columns=encoded_cols, index=test_df.index)

train_df = pd.concat([train_df.drop(columns=categorical_cols), train_encoded], axis=1)
val_df = pd.concat([val_df.drop(columns=categorical_cols), val_encoded], axis=1)
test_df = pd.concat([test_df.drop(columns=categorical_cols), test_encoded], axis=1)

numeric_cols = [
    col for col in df.select_dtypes(include=np.number)
    if df[col].nunique() > 2 and col != target_col
]

scaler = StandardScaler()
scaler.fit(train_df[numeric_cols])
train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_val = val_df.drop(columns=[target_col])
y_val = val_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Val accuracy:", model.score(X_val, y_val))
print("Test accuracy:", model.score(X_test, y_test))