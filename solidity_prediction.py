import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

secure_file_path = r"C:/Users/revku/Documents/KAGGLE DATASETS/solidity/BCCC-VolSCs-2023_Secure.csv"
vulnerable_file_path = r"C:/Users/revku/Documents/KAGGLE DATASETS/solidity/BCCC-VolSCs-2023_Vulnerable.csv"

try:
    secure_df = pd.read_csv(secure_file_path, encoding='utf-8')
    vulnerable_df = pd.read_csv(vulnerable_file_path, encoding='utf-8')
except FileNotFoundError:
    print("Error: One or both of the files were not found. Please check the file paths.")
    exit()
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit()

secure_df['label'] = 0
vulnerable_df['label'] = 1
df = pd.concat([secure_df, vulnerable_df], ignore_index=True)

if 'hash_id' not in df.columns:
    print("Error: The column 'hash_id' does not exist in the dataset. Please check the column names.")
    print("Available columns:", df.columns)
    exit()
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text) 
    return text.lower().strip() 

df['cleaned_code'] = df['hash_id'].apply(clean_text)
df.dropna(subset=['cleaned_code'], inplace=True)
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['cleaned_code'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

print(classification_report(y_test, y_pred))
def predict_fraud(input_text):
    if not input_text.strip():
        return "Invalid input: Empty contract code."
    cleaned_input = clean_text(input_text)
    input_vector = vectorizer.transform([cleaned_input])
    prediction = clf.predict(input_vector)
    return "Secure" if prediction[0] == 0 else "Vulnerable (Fraudulent)"

# Interactive testing
if __name__ == "__main__":
    while True:
        input_text = input("Enter the contract code to check for fraud (or 'exit' to quit): ")
        if input_text.lower() == 'exit':
            print("Exiting fraud detection system.")
            break
        print(predict_fraud(input_text))
