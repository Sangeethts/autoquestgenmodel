import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['Question Text'] = df['Question Text'].astype(str)
    return df

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    model = SVC()
    model.fit(X_train_vectorized, y_train)

    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def get_user_input():
    while True:
        num_questions_str = input("Enter the number of questions: ")
        if num_questions_str.isdigit():
            num_questions = int(num_questions_str)
            break
        else:
            print("Please enter a valid number.")
    
    blooms_level = input("Enter the Bloom's Taxonomy Level (e.g., Apply): ")
    difficulty_level = input("Enter the Difficulty Level (e.g., Medium): ")
    return num_questions, blooms_level, difficulty_level

def filter_dataset(df, blooms_level, difficulty_level):
    return df[(df["Bloom's Taxonomy Level"] == blooms_level) & (df["Difficulty Level"] == difficulty_level)]

def select_questions(filtered_df, num_questions):
    if len(filtered_df) < num_questions:
        print("Not enough questions available.")
        return None
    return filtered_df.sample(num_questions)

def save_model(model, file_path):
    joblib.dump(model, file_path)
    print("Model saved successfully.")

def main():
    file_path = "dataset.csv"
    df = load_data(file_path)
    df = preprocess_data(df)
    num_questions, blooms_level, difficulty_level = get_user_input()
    filtered_df = filter_dataset(df, blooms_level, difficulty_level)
    selected_questions = select_questions(filtered_df, num_questions)
    if selected_questions is not None:
        X = selected_questions['Question Text']
        y = selected_questions['Answer']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, vectorizer = train_model(X_train, y_train)
        evaluate_model(model, vectorizer, X_test, y_test)
        save_model(model, "trained_model.pkl")

if __name__ == "__main__":
    main()
