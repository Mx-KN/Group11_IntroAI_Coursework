from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import pandas as pd


def create_target(data):
    """
    Create a binary target column indicating if the next day's price moves up (1) or down (0).
    """
    print("Creating binary target column...")
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)  # Drop the last row as it won't have a valid target
    print("Binary target column created.")
    return data


def train_test_split_data(data, features):
    """
    Split data into training and testing sets.
    """
    print(f"Splitting data using features: {features}...")
    X = data[features]
    y = data['Target']

    # Split chronologically (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model with balanced class weighting.
    """
    print("Training Logistic Regression model with balanced class weighting...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model


def oversample_with_smote(X_train, y_train):
    """
    Apply SMOTE to oversample the minority class in the training data.
    """
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced class distribution: {pd.Series(y_balanced).value_counts()}")
    return X_balanced, y_balanced


def evaluate_classification_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate the classification model with an adjustable decision threshold.

    Parameters:
    - threshold: Decision threshold for classifying probabilities as 1 (default is 0.5).
    """
    print("Evaluating the model...")
    
    # Predict probabilities for class 1 (price increase)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Adjust threshold for classification
    print(f"Using decision threshold: {threshold}")
    y_pred = (y_pred_proba > threshold).astype(int)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred, accuracy, precision, recall


def run_classification(final_data, use_smote=False, threshold=0.5):
    """
    Execute logistic regression-based classification.

    Parameters:
    - use_smote: Whether to apply SMOTE for balancing.
    - threshold: Decision threshold for classification.
    """
    # Create binary target column
    print("\nCreating target column...")
    final_data = create_target(final_data)

    # Define features (exclude non-predictive columns)
    features = final_data.columns.difference(['Date', 'Close', 'Target'])
    print(f"Features for classification: {list(features)}")

    # Split the data
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split_data(final_data, features)

    # Optional: Apply SMOTE
    if use_smote:
        print("\nBalancing the training data using SMOTE...")
        X_train, y_train = oversample_with_smote(X_train, y_train)

    # Train the logistic regression model
    print("\nTraining logistic regression model...")
    model = train_logistic_regression(X_train, y_train)

    # Evaluate the model with a custom threshold
    print("\nEvaluating logistic regression model...")
    evaluate_classification_model(model, X_test, y_test, threshold=threshold)
