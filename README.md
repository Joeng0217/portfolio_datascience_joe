##  **Prediction of Customer Churn Using Machine Learning and Ensemble Techniques**
# Abstract
This project aims to predict customer churn in the telecommunications industry using various machine learning models, including Logistic Regression, Random Forest, and Support Vector Machine, alongside ensemble methods like XGBoost, AdaBoost, and Extra Trees Classifier. Key data preprocessing techniques such as SMOTE for class balancing and Principal Component Analysis (PCA) for dimensionality reduction were utilized. The Random Forest model optimized with Randomized Search CV achieved the highest accuracy of 86.10%, followed closely by XGBoost at 85.84%. The insights derived from this analysis support proactive customer retention strategies.

# Skills Demonstrated
Machine Learning Models: Logistic Regression, Random Forest, Support Vector Machine (SVM), AdaBoost, XGBoost, Extra Trees Classifier.
Data Preprocessing: Feature engineering, handling missing values, label encoding, one-hot encoding, and class balancing with SMOTE.
Dimensionality Reduction: PCA with variance preservation (95%).
Hyperparameter Tuning: Grid Search and Randomized Search CV for model optimization.
Model Validation: 10-fold and repeated K-fold cross-validation to assess model robustness.
Evaluation Metrics: ROC-AUC, F1-score, precision, recall, and confusion matrices.
Programming Libraries: NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, Plotly, Imbalanced-learn.

# Process Breakdown
**1. Dataset Overview**
- Source: Kaggle’s Telco Customer Churn Dataset (7043 instances, 21 features). 
- Features: Customer demographics, account details, and payment history (e.g., gender, tenure, total charges, monthly charges, contract type). 
- Target Variable: Churn (binary classification: churned vs. non-churned).

**2. Data Preprocessing**

- Class Balancing: Addressed 26.6% churn rate imbalance using **SMOTE**.
- Feature Encoding: Converted binary and categorical features using label and **one-hot encoding**.
- Scaling: Min-Max Scaling applied to normalize skewed numerical data.

**3. Exploratory Data Analysis (EDA)**
Insights:

- Higher monthly charges and shorter tenure correlate with increased churn.
- Non-churners are predominantly on two-year contracts with lower monthly charges.
- Churners commonly use electronic checks as their payment method.
- Visualizations: **Correlation heatmaps, boxplots, and distribution charts highlighted key trends**.

**4. Model Implementation**

- Baseline Models: Logistic Regression, Random Forest, SVM.
- Ensemble Models: AdaBoost, XGBoost, Extra Trees.
- Optimization: PCA preserved 95% variance; hyperparameter tuning improved model performance.
![image](https://github.com/user-attachments/assets/b4ca4b13-5847-4307-a01c-bd285b51ffb0)

**5. Model Validation and Tuning**
- Cross-Validation: Repeated 10-fold CV achieved the most robust validation scores.

**Hyperparameter Tuning:**

- Random Forest optimized with n_estimators=150, max_depth=50, and criterion='gini'.
- SVM fine-tuned with kernel=rbf, gamma=auto, and C=7.
- Logistic Regression optimized with solver=saga and C=1.0001.

**Key Insights**
**Feature Importance:**
Top predictors: Total Charges, Monthly Charges, Contract Type (month-to-month), Tenure, Online Security.

**Performance:**
RF with Randomized Search Hyperparameters (n_estimators= 150, min_samples_split= 10, min_samples_leaf= 4, max_features= 'log2', max_depth= 50, criterion= 'gini', bootstrap= True) achieved the highest accuracy of **86.10%**

**Conclusion**
This project demonstrates the application of advanced machine learning techniques and model optimization strategies to solve a real-world business problem. The resulting models provide actionable insights for targeted customer retention efforts, enabling telecommunications firms to minimize churn and maximize customer lifetime value.
