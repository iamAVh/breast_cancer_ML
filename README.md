# 🩺 Breast Cancer Detection using Machine Learning

This project implements a machine learning-based system to detect breast cancer using the **Wisconsin Breast Cancer Dataset**. The primary objective is to classify tumors as **benign** or **malignant** based on features extracted from breast cell images.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository  
- **Features:** Radius, texture, perimeter, area, smoothness, and more  
- **Target:**  
  - `0` → Malignant  
  - `1` → Benign  

---

## ✅ Project Overview

- Loads and preprocesses the breast cancer dataset  
- Performs **exploratory data analysis (EDA)** with correlation heatmaps and feature distributions  
- Splits data into training and test sets  
- Trains multiple machine learning models:  
  - Logistic Regression  
  - Random Forest  
  - Support Vector Machine (SVM)  
  - Decision Tree  
- Evaluates model accuracy and confusion matrix  
- Compares results to determine the best-performing model  
- Implements logistic regression from scratch to validate model results

---

## 📋 Project Details

1. **Data Acquisition and Cleaning**  
   The breast cancer detection dataset was obtained from online resources and underwent a thorough data cleaning process to ensure quality and consistency for analysis.

2. **Data Splitting**  
   The cleaned dataset was split into training and test sets to evaluate model performance on unseen data.

3. **Logistic Regression Model Using Scikit-learn**  
   - Initially, a Logistic Regression model was trained using all feature columns via the `scikit-learn` library.  
   - The model achieved an accuracy of **95.32%** on the test set.  
   - We generated the confusion matrix to evaluate classification performance and calculated the model's precision and recall based on this matrix.

4. **Feature Importance Analysis**  
   - Feature importance scores were computed to identify the most and least significant features affecting model performance.  
   - The dataset was divided into two subsets by excluding features as follows:  
     - Excluding the **5 most important features** resulted in an accuracy of **91.24%**.  
     - Excluding the **5 least important features** resulted in an accuracy of **94.16%**.

5. **Manual Logistic Regression Implementation (Without Scikit-learn)**  
   - Logistic regression was implemented from scratch by coding the sigmoid activation function, logistic loss function, gradient computation, and parameter (theta) updates.  
   - This approach achieved:  
     - **R² Score:** 0.7025  
     - **Accuracy:** 93.41%  
     - **Precision:** 90.50%

---

## 📈 Results

- Achieved over **96% accuracy** using Random Forest and SVM classifiers  
- Provided visual insights using Seaborn and Matplotlib

---

## 🧰 Technologies Used

- Python  
- Scikit-learn  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Jupyter Notebook  

---


