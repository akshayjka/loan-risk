import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def train_system():
    file_path = r'.\loan_approval.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ File not found at {file_path}")
        return

    # 1. Load and Sanitize
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    print("✅ Columns found in your file:", df.columns.tolist())

    # 2. Map your specific Target Variable
    # Your file uses 'loan_approved' instead of 'loan_status'
    if 'loan_approved' in df.columns:
        # If the data is already 0 and 1, we don't need to map it. 
        # If it's "Yes"/"No", uncomment the next line:
        df['loan_approved'] = df['loan_approved'].map({'Yes': 1, 'No': 0})
        y = df['loan_approved']
    else:
        print("❌ Error: Target column 'loan_approved' not found!")
        return

    # 3. Map your specific Features
    # Based on your output, we'll use the most relevant numeric fields
    feature_cols = ['income', 'loan_amount', 'credit_score', 'years_employed']
    
    print(f"🧠 Training on: {feature_cols}")
    X = df[feature_cols]

    # 4. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # 5. Save
    joblib.dump(model, 'loan_model.pkl')
    print("✅ Success! 'loan_model.pkl' created using your custom dataset.")

if __name__ == "__main__":
    train_system()

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(acc)