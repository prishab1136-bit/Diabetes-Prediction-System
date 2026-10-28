# Diabetes-Prediction-System
[readme_md.md](https://github.com/user-attachments/files/34094347/readme_md.md)
# Diabetes Prediction System Using Machine Learning

A machine learning-based diagnostic decision-support system built to predict early-stage diabetes onset using key physiological parameters from the Pima Indians Diabetes Dataset (`diabetes.csv`).

# Author Details

* **Author:** Prisha Biswas

* **Institution:** JSPM Rajarshi Shahu College of Pharmacy, Tathawade, Pune

* **Academic Year:** 2026–2027

# Overview & Objectives

Diabetes Mellitus is a chronic non-communicable metabolic disorder characterized by elevated blood glucose levels. Early detection enables timely lifestyle modifications and clinical intervention, drastically reducing the risk of long-term microvascular and macrovascular complications.

This project delivers an end-to-end predictive pipeline that ingests diagnostic metrics, resolves biological zero/missing values via median imputation, standardizes multi-scale physiological attributes using `StandardScaler`, and evaluates multiple supervised binary classification models to predict patient risk status ($0 =$ Non-Diabetic, $1 =$ Diabetic).

### Key Features:

* **Comprehensive Data Cleaning:** Identifies and imputes biologically invalid zero entries in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI.

* **Feature Normalization:** Uses Z-score standardization ($\mu=0, \sigma=1$) to optimize distance-sensitive classifiers.

* **Model Comparative Evaluation:** Benchmarks Support Vector Machine (SVM), Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN).

* **Interactive Predictive Engine:** Includes an execution routine to evaluate new patient clinical health readings.

## Dataset Schema (`diabetes.csv`)

The dataset comprises **768 female patient records** with 8 diagnostic predictor variables and 1 binary outcome label.

| Feature Name | Data Type | Range / Units | Clinical Relevance | 
| ----- | ----- | ----- | ----- | 
| **Pregnancies** | Integer | $0 - 17$ | Number of pregnancies | 
| **Glucose** | Numeric | $0 - 199$ mg/dL | Plasma glucose concentration (2-hour oral glucose tolerance test) | 
| **BloodPressure** | Numeric | $0 - 122$ mmHg | Diastolic blood pressure | 
| **SkinThickness** | Numeric | $0 - 99$ mm | Triceps skinfold thickness | 
| **Insulin** | Numeric | $0 - 846$ $\mu\text{IU/mL}$ | 2-Hour serum insulin | 
| **BMI** | Float | $0.0 - 67.1$ $\text{kg/m}^2$ | Body Mass Index | 
| **DiabetesPedigreeFunction** | Float | $0.078 - 2.42$ | Genetic risk score based on family history | 
| **Age** | Integer | $21 - 81$ years | Age in years | 
| **Outcome** | Integer | `0` or `1` | **Target Class:** $0=$ Non-Diabetic, $1=$ Diabetic | 

## Project Structure

```
├── diabetes.csv                       # Diagnostic healthcare dataset
├── Diabetes_Prediction_Notebook.ipynb # Step-by-step model training and evaluation notebook
├── project_report.pdf                 # Formatted academic project report
├── README.md                          # Comprehensive project documentation

```

## Methodology & Pipeline Workflow

1. **Data Ingestion & Profiling:** Ingest `diabetes.csv` and compute class balance distributions ($500$ Non-Diabetic / $268$ Diabetic).

2. **Missing/Zero Value Imputation:** Replace biologically impossible $0$ measurements across Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI with column medians:
   

   $$
   \text{Median}(X_j \setminus \{0\})
   $$

3. **Data Standardization:** Transform features onto a uniform scale using Z-score Normalization:
   

   $$
   z = \frac{x - \mu}{\sigma}
   $$

4. **Train-Test Stratified Split:** Divide data into 80% training ($614$ samples) and 20% testing ($154$ samples) with stratification.

5. **Model Evaluation:** Benchmark Support Vector Machine (Linear Kernel), Logistic Regression, Random Forest, and $K$-NN.

## Performance Benchmarks

Evaluated on the $154$-sample testing partition:

| Classifier | Train Accuracy | Test Accuracy | Precision | Recall (Sensitivity) | F1-Score | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Support Vector Machine (SVM)** | **78.6%** | **77.3%** | **0.73** | **0.65** | **0.69** | 
| **Logistic Regression** | 77.8% | 76.6% | 0.71 | 0.63 | 0.67 | 
| **Random Forest** | 100.0% | 76.0% | 0.69 | 0.61 | 0.65 | 
| **K-Nearest Neighbors (**$K=5$**)** | 81.2% | 74.0% | 0.65 | 0.57 | 0.61 | 

*Note: Support Vector Machine (SVM) demonstrated optimal generalization across test data with balanced precision-recall trade-offs and no overfitting.*

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed along with the required dependencies:

```
pip install numpy pandas scikit-learn matplotlib seaborn

```

### 2. Execution Code Sample

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_csv('diabetes.csv')

# 2. Impute Invalid Zeros with Median Values
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# 3. Separate Features and Target
X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

# 4. Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-Test Split & Model Training
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# 6. Run Patient Inference
sample_input = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
sample_array = scaler.transform(np.asarray(sample_input).reshape(1, -1))
prediction = model.predict(sample_array)

if prediction[0] == 0:
    print("Patient Status: NON-DIABETIC (Low Risk)")
else:
    print("Patient Status: DIABETIC (High Risk - Clinical Evaluation Advised)")

```

## Future Scope

* **Web & Mobile Interface Deployment:** Developing an interactive Streamlit or FastAPI web dashboard for healthcare providers.

* **Explainable AI (XAI):** Integrating SHAP and LIME to highlight key patient-specific clinical parameters driving each prediction.

* **EHR System Integration:** Connecting model execution pipelines with real-time electronic health record databases.

