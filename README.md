# 🚀 Credit Card Fraud Detection

[Test the project here !](https://fraud-detection-with-ai-frontend.streamlit.app/)

Detecting fraudulent credit card transactions using machine learning and deep learning.  
This project leverages the famous [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle to design, train, and evaluate models that can accurately spot fraud in highly imbalanced data.

---

## 🎯 Goals

1. **Explore & understand the dataset**  
   - Visualize PCA-transformed variables (`V1`–`V28`) + `Time` and `Amount`.  
   - Analyze the severe class imbalance (fraud ≈ 0.17% of all transactions).  

2. **Preprocess & rebalance**  
   - Normalize `Time` & `Amount`.  
   - Handle imbalance using **SMOTE**, **undersampling**, or **focal loss**.  

3. **Model training**  
   - Baseline models: **Logistic Regression and LightGBM**.  

4. **Hyperparameter optimization**  
   - Automated search with **Optuna**.  

5. **Evaluation & interpretation**  
   - Metrics: **AUC, Recall, F1-score, Precision-Recall curves**.  

---

## 🛠️ Tech Stack

- **Python 3.10+**  
- **Jupyter Notebooks** for experimentation  
- **Pandas / NumPy** → data manipulation  
- **Matplotlib / Seaborn** → visualization  
- **Scikit-learn** → baseline models & metrics  
- **LightGBM** → boosting methods  
- **Optuna** → hyperparameter optimization  

---

## 📊 Expected Results

- Significant reduction of **false negatives** (missed fraud cases).  
- 91% of precision 

---

## 🌟 Why This Project Stands Out

✅ Tackles a **real-world problem** faced by banks & fintechs.   
✅ Includes **automated hyperparameter tuning (Optuna)**.  
✅ This is a great project for progressing.
✅ I integrated a frontend with **Streamlit**.

---

> 💡 This project is designed to be **recruiter-friendly**: it highlights strong ML/AI skills and the ability to build interpretable, production-ready models.  
---
