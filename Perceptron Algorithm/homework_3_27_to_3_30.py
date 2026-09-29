"""
Machine Learning - Perceptron Learning Algorithm
Exercises 3.27 to 3.30

Author: ManhVu
Date: 2026
Description: Complete solutions for Perceptron exercises from Machine Learning course
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from typing import Tuple, Dict, List


# ============================================================================
# PERCEPTRON CLASS - Used for all exercises
# ============================================================================

class Perceptron:
    """
    Perceptron classifier using the perceptron learning algorithm.
    
    Attributes:
        learning_rate (float): Learning rate for weight updates (default: 0.01)
        epochs (int): Number of passes over the training data (default: 100)
        weights (np.ndarray): Weight vector after fitting
        bias (float): Bias term after fitting
        errors (list): Number of misclassifications in each epoch
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        """Initialize the Perceptron."""
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.errors = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        """Train the perceptron on the given data."""
        y_converted = np.where(y == 0, -1, 1)
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = 0
        
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y_converted):
                prediction = self._predict_single(xi)
                error = target - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += 1
            
            self.errors.append(errors)
            if errors == 0:
                break
        
        return self
    
    def _predict_single(self, x: np.ndarray) -> int:
        """Predict the class label for a single sample."""
        score = np.dot(x, self.weights) + self.bias
        return 1 if score > 0 else -1
    
    def _predict_score(self, x: np.ndarray) -> float:
        """Get the raw score for a single sample."""
        return np.dot(x, self.weights) + self.bias
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        predictions = np.array([self._predict_single(xi) for xi in X])
        return np.where(predictions == -1, 0, 1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction scores."""
        scores = np.array([self._predict_score(xi) for xi in X])
        return scores
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def get_weights(self) -> Tuple[np.ndarray, float]:
        """Get the learned weights and bias."""
        return self.weights, self.bias


# ============================================================================
# EXERCISE 3.27: Basic Perceptron Prediction
# ============================================================================

def exercise_3_27():
    """
    Exercise 3.27: Calculate w^T x and check misclassification
    
    Given:
        w = [1, 2, -10]^T
        x = [3, 4, 1]^T
        y = -1 (true label)
    
    Tasks:
        1. Calculate w^T x
        2. Determine predicted label
        3. Check if misclassified
    """
    print("\n" + "=" * 70)
    print("EXERCISE 3.27: PERCEPTRON PREDICTION")
    print("=" * 70)
    
    w = np.array([1, 2, -10])
    x = np.array([3, 4, 1])
    y_true = -1
    
    print(f"\nGiven:")
    print(f"  w = {w}")
    print(f"  x = {x}")
    print(f"  y (true label) = {y_true}")
    
    # 1. Calculate w^T x
    print(f"\n1. Calculate w^T x:")
    score = np.dot(w, x)
    print(f"   w^T x = {w[0]}×{x[0]} + {w[1]}×{x[1]} + {w[2]}×{x[2]}")
    print(f"   w^T x = {w[0]*x[0]} + {w[1]*x[1]} + {w[2]*x[2]}")
    print(f"   w^T x = {score}")
    
    # 2. Determine predicted label
    print(f"\n2. Determine predicted label:")
    y_pred = 1 if score > 0 else -1
    print(f"   w^T x = {score} {'>' if score > 0 else '<'} 0")
    print(f"   → Predicted label ŷ = {y_pred}")
    
    # 3. Check misclassification
    print(f"\n3. Check misclassification:")
    is_misclassified = (y_pred != y_true)
    print(f"   Predicted ŷ = {y_pred}")
    print(f"   True y = {y_true}")
    print(f"   ŷ ≠ y? {is_misclassified}")
    
    if is_misclassified:
        print(f"   → Data point IS MISCLASSIFIED ✗")
    else:
        print(f"   → Data point is CORRECTLY CLASSIFIED ✓")
    
    print("\n" + "=" * 70)
    
    return {
        'w_T_x': score,
        'predicted_label': y_pred,
        'true_label': y_true,
        'misclassified': is_misclassified
    }


# ============================================================================
# EXERCISE 3.28: Perceptron Update
# ============================================================================

def exercise_3_28():
    """
    Exercise 3.28: Perceptron weight update
    
    Given:
        w = [-2, 1, 0]^T
        x = [2, 3, 1]^T
        y = 1 (true label)
    
    Tasks:
        1. Check if misclassified
        2. Perform Perceptron update (if needed)
        3. Recalculate w^T x after update
    """
    print("\n" + "=" * 70)
    print("EXERCISE 3.28: PERCEPTRON UPDATE STEP")
    print("=" * 70)
    
    w_old = np.array([-2, 1, 0])
    x = np.array([2, 3, 1])
    y_true = 1
    
    print(f"\nGiven:")
    print(f"  w (initial) = {w_old}")
    print(f"  x = {x}")
    print(f"  y (true label) = {y_true}")
    
    # ========== STEP 1: Check misclassification ==========
    print(f"\n{'─' * 70}")
    print("STEP 1: Check misclassification")
    print(f"{'─' * 70}")
    
    score = np.dot(w_old, x)
    y_pred = 1 if score > 0 else -1
    is_misclassified = (y_pred != y_true)
    
    print(f"\nCalculate w^T x:")
    print(f"  w^T x = ({w_old[0]})×({x[0]}) + ({w_old[1]})×({x[1]}) + ({w_old[2]})×({x[2]})")
    print(f"  w^T x = {w_old[0]*x[0]} + {w_old[1]*x[1]} + {w_old[2]*x[2]}")
    print(f"  w^T x = {score}")
    
    print(f"\nDetermine predicted label:")
    print(f"  w^T x = {score} {'>' if score > 0 else '<'} 0")
    print(f"  → Predicted label ŷ = {y_pred}")
    
    print(f"\nCheck result:")
    print(f"  Predicted ŷ = {y_pred}")
    print(f"  True y = {y_true}")
    if is_misclassified:
        print(f"  → Data point IS MISCLASSIFIED ✗")
    else:
        print(f"  → Data point is CORRECTLY CLASSIFIED ✓")
    
    # ========== STEP 2: Update weights ==========
    print(f"\n{'─' * 70}")
    print("STEP 2: Perceptron weight update")
    print(f"{'─' * 70}")
    
    if is_misclassified:
        print(f"\nUpdate rule: w_new = w_old + y × x")
        print(f"w_new = {w_old} + {y_true} × {x}")
        
        w_new = w_old + y_true * x
        
        print(f"\nDetailed calculation:")
        print(f"  w_new[0] = {w_old[0]} + {y_true} × {x[0]} = {w_old[0]} + {y_true * x[0]} = {w_new[0]}")
        print(f"  w_new[1] = {w_old[1]} + {y_true} × {x[1]} = {w_old[1]} + {y_true * x[1]} = {w_new[1]}")
        print(f"  w_new[2] = {w_old[2]} + {y_true} × {x[2]} = {w_old[2]} + {y_true * x[2]} = {w_new[2]}")
        
        print(f"\n→ w_new = {w_new}")
    else:
        print(f"\nNo update needed (correctly classified)")
        w_new = w_old
    
    # ========== STEP 3: Recalculate w^T x ==========
    print(f"\n{'─' * 70}")
    print("STEP 3: Recalculate w^T x after update")
    print(f"{'─' * 70}")
    
    score_new = np.dot(w_new, x)
    y_pred_new = 1 if score_new > 0 else -1
    is_correct_new = (y_pred_new == y_true)
    
    print(f"\nAfter update:")
    print(f"  w_new = {w_new}")
    print(f"  x = {x}")
    
    print(f"\nCalculate w_new^T x:")
    print(f"  w_new^T x = ({w_new[0]})×({x[0]}) + ({w_new[1]})×({x[1]}) + ({w_new[2]})×({x[2]})")
    print(f"  w_new^T x = {w_new[0]*x[0]} + {w_new[1]*x[1]} + {w_new[2]*x[2]}")
    print(f"  w_new^T x = {score_new}")
    
    print(f"\nPredicted label:")
    print(f"  w_new^T x = {score_new} {'>' if score_new > 0 else '<'} 0")
    print(f"  → Predicted label ŷ = {y_pred_new}")
    
    print(f"\nVerification:")
    print(f"  Predicted ŷ = {y_pred_new}")
    print(f"  True y = {y_true}")
    if is_correct_new:
        print(f"  → Data point is NOW CORRECTLY CLASSIFIED ✓")
    else:
        print(f"  → Still MISCLASSIFIED ✗")
    
    print("\n" + "=" * 70)
    
    return {
        'w_initial': w_old,
        'w_updated': w_new,
        'initial_score': score,
        'updated_score': score_new,
        'initial_prediction': y_pred,
        'updated_prediction': y_pred_new,
        'was_misclassified': is_misclassified,
        'correct_after_update': is_correct_new
    }


# ============================================================================
# EXERCISE 3.29: Perceptron Class
# ============================================================================

def exercise_3_29():
    """
    Exercise 3.29: Perceptron class implementation
    
    Create a Perceptron class with:
    - fit() method for training
    - predict() method for prediction
    - Demonstration on synthetic data
    """
    print("\n" + "=" * 70)
    print("EXERCISE 3.29: PERCEPTRON CLASS IMPLEMENTATION")
    print("=" * 70)
    
    print("\n✓ Perceptron class has been created")
    print("  Features:")
    print("    • fit(X, y) - Train the model")
    print("    • predict(X) - Predict labels")
    print("    • score(X, y) - Calculate accuracy")
    print("    • get_weights() - Get learned weights")
    
    # Demonstration
    print("\n" + "─" * 70)
    print("DEMONSTRATION ON SYNTHETIC DATA")
    print("─" * 70)
    
    # Create synthetic data
    np.random.seed(42)
    
    # Class +1
    class1 = np.random.randn(50, 2) + [3, 3]
    
    # Class -1
    class2 = np.random.randn(50, 2) + [0, 0]
    
    # Combine and shuffle
    X = np.vstack([class1, class2])
    y = np.hstack([np.ones(50), np.zeros(50)])
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nDataset:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Train
    print(f"\nTraining Perceptron...")
    perceptron = Perceptron(learning_rate=0.01, epochs=100)
    perceptron.fit(X_train, y_train)
    
    # Get weights
    weights, bias = perceptron.get_weights()
    print(f"  ✓ Training completed")
    print(f"  Weights: {weights}")
    print(f"  Bias: {bias}")
    
    # Evaluate
    train_acc = perceptron.score(X_train, y_train)
    test_acc = perceptron.score(X_test, y_test)
    
    print(f"\nAccuracy:")
    print(f"  Training: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Predict new samples
    print(f"\nPredictions on new samples:")
    new_samples = np.array([
        [3.5, 3.5],
        [-0.5, -0.5],
        [2.0, 2.0]
    ])
    
    predictions = perceptron.predict(new_samples)
    for i, (sample, pred) in enumerate(zip(new_samples, predictions)):
        print(f"  Sample {i+1}: {sample} → Predicted label: {int(pred)}")
    
    print("\n" + "=" * 70)
    
    return {
        'perceptron': perceptron,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    }


# ============================================================================
# EXERCISE 3.30: Binary Classification on Real Dataset
# ============================================================================

def exercise_3_30():
    """
    Exercise 3.30: Binary classification using Perceptron
    
    Dataset: Breast Cancer (Malignant vs Benign)
    Evaluate using: Accuracy, Precision, Recall, F1-Score
    """
    print("\n" + "=" * 70)
    print("EXERCISE 3.30: BINARY CLASSIFICATION - REAL DATASET")
    print("=" * 70)
    
    # Load dataset
    print("\n1. LOADING DATASET")
    print("─" * 70)
    
    cancer_data = load_breast_cancer()
    X = cancer_data.data
    y = cancer_data.target
    
    print(f"Dataset: Breast Cancer Classification")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: 2 (Binary)")
    
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        label = "Malignant (0)" if cls == 0 else "Benign (1)"
        percentage = 100 * count / len(y)
        print(f"    Class {cls} ({label}): {count} ({percentage:.1f}%)")
    
    # Preprocessing
    print("\n2. DATA PREPROCESSING")
    print("─" * 70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Features standardized (StandardScaler)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Training
    print("\n3. TRAINING PERCEPTRON")
    print("─" * 70)
    
    perceptron = Perceptron(learning_rate=0.01, epochs=100)
    print("Training in progress...")
    perceptron.fit(X_train, y_train)
    print(f"✓ Training completed")
    
    # Predictions
    print("\n4. MAKING PREDICTIONS")
    print("─" * 70)
    
    y_train_pred = perceptron.predict(X_train)
    y_test_pred = perceptron.predict(X_test)
    
    print(f"✓ Predictions on training set: {len(y_train_pred)} samples")
    print(f"✓ Predictions on test set: {len(y_test_pred)} samples")
    
    # Evaluation
    print("\n5. MODEL EVALUATION")
    print("=" * 70)
    
    # Training metrics
    train_metrics = {
        'Accuracy': accuracy_score(y_train, y_train_pred),
        'Precision': precision_score(y_train, y_train_pred),
        'Recall': recall_score(y_train, y_train_pred),
        'F1-Score': f1_score(y_train, y_train_pred)
    }
    
    # Test metrics
    test_metrics = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1-Score': f1_score(y_test, y_test_pred)
    }
    
    print("\nTRAINING SET METRICS:")
    for metric, value in train_metrics.items():
        print(f"  {metric:.<30} {value:.4f}")
    
    print("\nTEST SET METRICS:")
    for metric, value in test_metrics.items():
        print(f"  {metric:.<30} {value:.4f}")
    
    # Confusion Matrix
    print("\n6. CONFUSION MATRIX (Test Set)")
    print("─" * 70)
    
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"True Negatives (TN):  {cm[0, 0]}")
    print(f"False Positives (FP): {cm[0, 1]}")
    print(f"False Negatives (FN): {cm[1, 0]}")
    print(f"True Positives (TP):  {cm[1, 1]}")
    
    # Classification Report
    print("\n7. CLASSIFICATION REPORT (Test Set)")
    print("─" * 70)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Malignant', 'Benign']))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['Accuracy']:.4f} ({test_metrics['Accuracy']*100:.2f}%)")
    print(f"  Precision: {test_metrics['Precision']:.4f}")
    print(f"  Recall:    {test_metrics['Recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['F1-Score']:.4f}")
    
    print(f"\nModel Interpretation:")
    print(f"  • {test_metrics['Accuracy']*100:.2f}% of predictions are correct")
    print(f"  • {test_metrics['Precision']*100:.2f}% of predicted positive are actually positive")
    print(f"  • {test_metrics['Recall']*100:.2f}% of actual positive are correctly identified")
    
    print("\n" + "=" * 70)
    
    return {
        'perceptron': perceptron,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'confusion_matrix': cm
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MACHINE LEARNING - PERCEPTRON LEARNING ALGORITHM")
    print("Exercises 3.27 to 3.30")
    print("=" * 70)
    
    # Run all exercises
    result_3_27 = exercise_3_27()
    result_3_28 = exercise_3_28()
    result_3_29 = exercise_3_29()
    result_3_30 = exercise_3_30()
    
    # Final Summary
    print("\n" + "=" * 70)
    print("ALL EXERCISES COMPLETED ✓")
    print("=" * 70)
    
    print("\nExercise Summary:")
    print("  ✓ 3.27: Perceptron prediction and misclassification check")
    print("  ✓ 3.28: Perceptron weight update step")
    print("  ✓ 3.29: Perceptron class implementation with fit/predict")
    print("  ✓ 3.30: Binary classification on real dataset")
    print("           - Accuracy:  " + f"{result_3_30['test_metrics']['Accuracy']:.4f}")
    print("           - Precision: " + f"{result_3_30['test_metrics']['Precision']:.4f}")
    print("           - Recall:    " + f"{result_3_30['test_metrics']['Recall']:.4f}")
    print("           - F1-Score:  " + f"{result_3_30['test_metrics']['F1-Score']:.4f}")
    
    print("\n" + "=" * 70 + "\n")
