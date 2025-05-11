import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

MAIN_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
DATA_DIR = os.path.join(MAIN_DIR, "extracted_data")
MODEL_DIR = os.path.join(MAIN_DIR, "trained_model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "StegGuard_Random_Forest_Classifier"
RANDOM_STATE = 42
N_ESTIMATORS = 100
CLASS_NAMES = ["Clean", "Stego"]

def load_data(data_dir):
    """Load training, validation, and test data from pickle files"""
    try:
        X_train, y_train = joblib.load(os.path.join(data_dir, "train_data.pkl"))
        X_val, y_val = joblib.load(os.path.join(data_dir, "val_data.pkl"))
        X_test, y_test = joblib.load(os.path.join(data_dir, "test_data.pkl"))
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}") from e

def train_model(X_train, y_train):
    """Train and return a Random Forest classifier"""
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X, y, set_name):
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=CLASS_NAMES)
    conf_matrix = confusion_matrix(y, y_pred)
    
    print(f"\n{set_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    return accuracy, report, conf_matrix

def save_model(model, model_dir, model_name):
    """Save model"""
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump({
        'model': model,
        'class_names': CLASS_NAMES,
        'random_state': RANDOM_STATE,
        'n_estimators': N_ESTIMATORS
    }, model_path)

if __name__ == "__main__":
    try:
        print("Loading data...")
        train_data, val_data, test_data = load_data(DATA_DIR)
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        print("Training model...")
        model = train_model(X_train, y_train)

        train_accuracy, train_report, train_conf_matrix = evaluate_model(model, X_train, y_train, "Training")
        val_accuracy, val_report, val_conf_matrix = evaluate_model(model, X_val, y_val, "Validation")
        test_accuracy, test_report, test_conf_matrix = evaluate_model(model, X_test, y_test, "Testing")

        save_model(model, MODEL_DIR, MODEL_NAME)
        print(f"\nModel and artifacts saved in '{MODEL_DIR}' directory")

    except Exception as e:
        print(f"Error occurred: {str(e)}")