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

# # **Project Title: Hotel Reservation Cancellation Prediction Using Data Analysis and Statistical Techniques in SAS**
**Abstract**
This project investigates the factors influencing hotel reservation cancellations using descriptive and inferential statistical methods in SAS. By exploring key attributes such as lead time, average price, and customer behavior, this study provides actionable insights for hotel management to optimize pricing strategies, improve resource allocation, and enhance customer retention. Key analyses include data preprocessing, exploratory data analysis (EDA), feature engineering, and hypothesis testing. Results highlight that longer lead times and higher average booking prices significantly increase cancellation rates.

**Skills Demonstrated**
- Data Management: Comprehensive data cleaning and handling of missing values, outliers, and inconsistent entries using SAS.
- Descriptive Analytics: Summarizing data trends and distributions using histograms, boxplots, and frequency tables.
- Inferential Statistics: Performing hypothesis testing (T-tests, ANOVA) to identify significant factors affecting cancellations.
- Feature Engineering: Creating derived variables (e.g., total nights, total customers) for enhanced analysis.
- Correlation Analysis: Exploring relationships between variables using correlation coefficients and scatterplots.
- Visualization: Using SAS-generated charts (e.g., bar charts, panel plots) to present trends and patterns.

**Dataset Overview**
- Source: Kaggle’s Hotel Booking Demand Dataset.
- Size: 36,285 records with 17 attributes.

**Features:**
- Categorical: Booking status, room type, market segment, meal type.
- Numerical: Lead time, average price, number of nights, and special requests.
- Target Variable: Booking status (canceled or not canceled).

**Process Breakdown**
**1. Data Preprocessing**
- Handling Missing Values:
- Replaced categorical missing values with the mode (e.g., "Online" for market segment).
- Removed rows with missing reservation dates due to their minimal impact (<0.02%).
- 
**Outlier Treatment:**
- Identified and removed outliers in numerical variables like average price using IQR thresholds.

**Feature Engineering:**
- Created Exploratory Data Analysis (EDA)
  
**Key Insights:**

- Most bookings involved 2 adults and lasted 2-3 nights.
- Lead Time: Canceled bookings had a mean lead time of 137 days, while non-canceled bookings averaged 59 days.
- Average Price: Higher-priced bookings (> $100/day) had a significantly higher cancellation rate (40% vs. 24.5% for lower prices).
- Market Segment: Online bookings showed the highest cancellation rate (37.56%), while offline and complementary bookings had lower rates.

**Visualization:**

- Boxplots and histograms revealed that canceled bookings were skewed toward higher lead times and prices.
- Bar charts highlighted cancellation trends across price categories and market segments. new variables such as:
  a. Total Nights: Sum of weekend and weekday nights.
  b. Total Customers: Sum of adults and children per booking.
  c. ADR (Average Daily Rate): Average price divided by total nights.

**Statistical Analysis**
- T-Test: Lead times for canceled bookings were significantly longer than non-canceled bookings (mean 137.5 days vs. 59.05 days, p < 0.0001).
- ANOVA: Higher room prices and certain room types were significantly associated with cancellations (p < 0.0001). Booking status explained 2.11% of price variability.
- Correlation: Moderate positive correlation between lead time and cancellations (R = 0.43). Weak correlation between special requests and booking status.

**Key Findings**
- Lead Time: Bookings with longer lead times (>100 days) were more likely to be canceled.
- Average Price: Higher-priced bookings (> $100/day) had a 40% cancellation rate, compared to 24.5% for lower-priced bookings.
- Market Segment: Online bookings had the highest cancellation rate (37.56%).
- Seasonality: Cancellations peaked in Q3 (July–September), driven by speculative bookings and peak travel demand.

**Recommendations**
- Dynamic Pricing: Introduce early-bird discounts for longer lead times and adjust prices during peak seasons to minimize speculative bookings.
- Targeted Marketing: Focus on online customers with personalized offers and incentives to reduce cancellations.
- Flexible Policies: Offer tiered cancellation policies based on lead time and price to retain high-value customers.
- Loyalty Programs: Strengthen repeat-customer incentives for corporate and complementary market segments.

**Conclusion**
This project highlights critical factors influencing hotel reservation cancellations, including lead time, pricing, and customer behavior. Statistical analysis revealed actionable insights to optimize revenue management, improve booking stability, and enhance customer retention strategies. These findings can guide data-driven decision-making in the hospitality industry, promoting more effective operational planning and resource allocation.






