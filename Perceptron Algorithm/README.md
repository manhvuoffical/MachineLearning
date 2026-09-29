# Machine Learning - Perceptron Learning Algorithm

**Exercises 3.27 to 3.30**

Comprehensive solutions for Perceptron exercises from Machine Learning course.

---

## 📋 Exercises Overview

### **Exercise 3.27**: Perceptron Prediction
✓ Calculate w^T x  
✓ Determine predicted label  
✓ Check misclassification  

**Given:**
- w = [1, 2, -10]^T
- x = [3, 4, 1]^T
- y = -1 (true label)

**Result:** Data point is MISCLASSIFIED

---

### **Exercise 3.28**: Perceptron Weight Update
✓ Check if misclassified  
✓ Perform Perceptron update (if needed)  
✓ Recalculate w^T x after update  

**Given:**
- w = [-2, 1, 0]^T
- x = [2, 3, 1]^T
- y = 1 (true label)

**Result:** 
- Initial w^T x = -1 (misclassified)
- Updated w = [0, 4, 1]
- New w^T x = 13 (correctly classified) ✓

---

### **Exercise 3.29**: Perceptron Class Implementation
✓ Implement Perceptron class with fit() and predict()  
✓ Support for binary classification  
✓ Early stopping on convergence  
✓ Accuracy calculation  

**Features:**
```python
perceptron = Perceptron(learning_rate=0.01, epochs=100)
perceptron.fit(X_train, y_train)
predictions = perceptron.predict(X_test)
accuracy = perceptron.score(X_test, y_test)
```

**Demo Results:**
- Training Accuracy: 100.00%
- Test Accuracy: 100.00%

---

### **Exercise 3.30**: Binary Classification on Real Dataset
✓ Dataset: Breast Cancer (569 samples, 30 features)  
✓ Binary classification: Malignant vs Benign  
✓ Comprehensive evaluation: Accuracy, Precision, Recall, F1-Score  
✓ Confusion matrix and classification report  

**Test Set Performance:**

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.37% |
| **Precision** | 98.59% |
| **Recall** | 97.22% |
| **F1-Score** | 0.9790 |

**Class Performance:**
- **Malignant (Class 0):** Precision 95%, Recall 98%
- **Benign (Class 1):** Precision 99%, Recall 97%

**Confusion Matrix:**
```
              Predicted
           Malignant Benign
Actual
Malignant      41       1
Benign          2      70
```

---

## 📁 Files

```
machine-learning-homework/
├── homework_3_27_to_3_30.py    (Main solution - all exercises)
├── requirements.txt             (Python dependencies)
├── README.md                    (This file)
└── outputs/                     (Visualization plots)
    ├── training_curve.png
    ├── metrics_comparison.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── class_distribution.png
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Exercises
```bash
python homework_3_27_to_3_30.py
```

This will execute all 4 exercises and display detailed results for each.

### 3. Output
The program will print:
- Exercise 3.27: Prediction and misclassification check
- Exercise 3.28: Weight update and recalculation
- Exercise 3.29: Perceptron class demo on synthetic data
- Exercise 3.30: Binary classification metrics and results

---

## 📊 Key Results

### Exercise 3.30 Analysis

**Model Performance:**
- **Excellent Accuracy (97.37%):** Model correctly classifies 97 out of 100 samples
- **High Precision (98.59%):** Only 1.41% false alarm rate for positive predictions
- **Strong Recall (97.22%):** Catches 97.22% of actual positive cases (important for medical diagnosis)
- **High F1-Score (0.9790):** Good balance between precision and recall

**Why This Matters:**
- In medical diagnosis, high recall means catching most actual cancer cases
- High precision means minimizing unnecessary biopsies
- This Perceptron model achieves a good balance for this use case

---

## 💡 Implementation Highlights

### Perceptron Class Features
```python
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100)
    def fit(X, y)           # Training
    def predict(X)          # Prediction
    def score(X, y)         # Accuracy
    def get_weights()       # Get learned parameters
```

### Key Implementation Details
1. **Standardization:** Features scaled using StandardScaler
2. **Label Conversion:** 0/-1 for training, 0/1 for output
3. **Early Stopping:** Converges when no errors in epoch
4. **Random Seed:** Set for reproducibility

---

## 📈 Dataset Information

**Breast Cancer Dataset:**
- **Source:** sklearn.datasets.load_breast_cancer
- **Samples:** 569
- **Features:** 30 (computed from digitized images)
- **Target:** Binary (0: Malignant, 1: Benign)
- **Class Distribution:** 37.3% Malignant, 62.7% Benign

**Data Split:**
- Training: 455 samples (80%)
- Test: 114 samples (20%)

---

## 🔧 Technical Details

### Algorithm: Perceptron Learning Rule
```
w_new = w_old + learning_rate × error × x
bias_new = bias_old + learning_rate × error
```

Where:
- error = y_true - y_predicted
- Update only when misclassification occurs

### Hyperparameters Used
- Learning Rate (η): 0.01
- Max Epochs: 100
- Early Stopping: When error = 0

### Data Preprocessing
- Standardization: (x - mean) / std
- Train-test split: 80-20 with stratification

---

## 📝 Author

**ManhVu**  
Computer Science Student  
Vietnam

---

## 📅 Date

2026

---

## 🎓 Course Information

**Course:** Machine Learning  
**Topic:** Perceptron Learning Algorithm  
**Difficulty:** Beginner to Intermediate  
**Duration:** ~2-3 hours

---

## ✅ Checklist

- [x] Exercise 3.27: Perceptron prediction
- [x] Exercise 3.28: Weight update
- [x] Exercise 3.29: Class implementation
- [x] Exercise 3.30: Real-world application
- [x] Evaluation metrics
- [x] Documentation
- [x] Code comments

---

## 📚 References

1. Perceptron Learning Algorithm: https://en.wikipedia.org/wiki/Perceptron
2. Scikit-learn Documentation: https://scikit-learn.org
3. Breast Cancer Dataset: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin

---

## 🤝 Contributing

Feel free to fork, modify, and improve this solution!

---

**Happy Learning!** 🚀
