import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Function to load test data
def load_test_data(file_path):
    return pd.read_csv(file_path)

# Function to evaluate the model
def evaluate_saved_model(model, vectorizer, X_test, y_test):
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# Load the saved model
def load_model(file_path):
    return joblib.load(file_path)

def main():
    # Load test data
    test_file_path = "test_data.csv"
    test_df = load_test_data(test_file_path)
    
    # Load saved model
    model_file_path = "trained_model.pkl"
    saved_model = load_model(model_file_path)
    
    # Perform evaluation
    X_test = test_df['Question Text']
    y_test = test_df['Answer']
    vectorizer = joblib.load("vectorizer.pkl")
    evaluate_saved_model(saved_model, vectorizer, X_test, y_test)
    
    # Additional tasks...
    # For example: deploy model, update documentation, communicate results, etc.

if __name__ == "__main__":
    main()
