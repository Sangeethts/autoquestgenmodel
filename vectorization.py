import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Function to load test data
def load_test_data(file_path):
    return pd.read_csv(file_path)

# Function to save the vectorizer
def save_vectorizer(vectorizer, file_path):
    joblib.dump(vectorizer, file_path)
    print("Vectorizer saved successfully.")

# Function to load the vectorizer
def load_vectorizer(file_path):
    return joblib.load(file_path)

# Function to vectorize text data
def vectorize_text(text_data, vectorizer):
    return vectorizer.transform(text_data)

# Function to train the model and save both model and vectorizer
def train_model_and_save(X_train, y_train, vectorizer_file_path, model_file_path):
    # Vectorize the training data
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Save the vectorizer
    save_vectorizer(vectorizer, vectorizer_file_path)

    # Handle missing values in X_train_vectorized
    imputer = SimpleImputer(strategy='mean')
    X_train_vectorized = imputer.fit_transform(X_train_vectorized)

    # Split data into training and validation sets
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_vectorized, y_train, test_size=0.2, random_state=42)

    # Train the model (using RandomForestClassifier as an example)
    model = RandomForestClassifier()
    model.fit(X_train_split, y_train_split)

    # Save the model
    joblib.dump(model, model_file_path)
    print("Model saved successfully.")

# Function to load the model
def load_model(file_path):
    return joblib.load(file_path)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def main():
    # Load data
    dataset_file_path = "dataset.csv"
    df = pd.read_csv(dataset_file_path)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Split data into features and target
    X = df['Question Text']
    y = df['Answer']

    # Train the model and save
    train_model_and_save(X, y, "vectorizer.pkl", "trained_model.pkl")

    # Load test data
    test_file_path = "dataset.csv"
    test_df = load_test_data(test_file_path)

    # Load the vectorizer
    vectorizer = load_vectorizer("vectorizer.pkl")

    # Vectorize the test data
    X_test_vectorized = vectorize_text(test_df['Question Text'], vectorizer)

    # Load the model
    model = load_model("trained_model.pkl")

    # Evaluate the model
    evaluate_model(model, X_test_vectorized, test_df['Answer'])

if __name__ == "__main__":
    main()
