##  **Prediction of Customer Churn Using Machine Learning and Ensemble Techniques**
# Abstract
This project aims to predict customer churn in the telecommunications industry using various machine learning models, including Logistic Regression, Random Forest, and Support Vector Machine, alongside ensemble methods like XGBoost, AdaBoost, and Extra Trees Classifier. Key data preprocessing techniques such as SMOTE for class balancing and Principal Component Analysis (PCA) for dimensionality reduction were utilized. The Random Forest model optimized with Randomized Search CV achieved the highest accuracy of 86.10%, followed closely by XGBoost at 85.84%. The insights derived from this analysis support proactive customer retention strategies.

# Skills Demonstrated
- Machine Learning Models: Logistic Regression, Random Forest, Support Vector Machine (SVM), AdaBoost, XGBoost, Extra Trees Classifier.
- Data Preprocessing: Feature engineering, handling missing values, label encoding, one-hot encoding, and class balancing with SMOTE.
- Dimensionality Reduction: PCA with variance preservation (95%).
- Hyperparameter Tuning: Grid Search and Randomized Search CV for model optimization.
- Model Validation: 10-fold and repeated K-fold cross-validation to assess model robustness.
- Evaluation Metrics: ROC-AUC, F1-score, precision, recall, and confusion matrices.
- Programming Libraries: NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, Plotly, Imbalanced-learn.

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

# **Project Title: Customer Segmentation for Retention Strategies in the Banking Sector**

**Abstract**
This project explores customer segmentation to address churn and retention in the banking industry, using clustering and decision tree models implemented in SAS Enterprise Miner. By analyzing a Kaggle dataset with 22 features and 10,127 rows, the study identifies distinct customer profiles among existing and churned customers. The findings enable targeted retention strategies, emphasizing customer engagement and personalized incentives to reduce attrition and enhance profitability.

**Dataset Overview**
Source: Kaggle dataset on credit card customer attrition.

**Key Features:**
- Demographics: Age, gender, marital status, education level.
- Behavioral Metrics: Total transactions, transaction amounts, inactivity periods.
- Financial Metrics: Credit limit, average open-to-buy, revolving balance, utilization ratio.
  
**Target Groups:**
- Existing customers (retained).
- Attrite customers (churned).

**Skills Demonstrated**
- Data Cleaning & Preprocessing: Handled skewed data, standardized variables, and applied dummy encoding to categorical variables for clustering.

**Exploratory Data Analysis (EDA):**
- Univariate analysis to examine distributions (e.g., customer age, credit utilization).
- Identification of key trends (e.g., churned customers had lower transaction counts).
  
**Clustering Models:**
- Determined optimal cluster numbers using misclassification rates.
- Segmented existing customers into 4 clusters and churned customers into 3 clusters.
- Decision Tree Modeling: Validated cluster assignments with a hybrid approach, achieving misclassification rates below 0.5%.
- Feature Engineering: Ranked variable importance for segmentation (e.g., credit limit, total transactions, income category).
- Statistical Insights: Highlighted differences between clusters, such as higher credit limits in high-value segments and lower utilization in churn-prone groups.

**Process Breakdown**

**1. Data Preprocessing**
- Identified and retained features relevant for clustering.
- Handled skewed variables (e.g., total transaction amount) using log transformation.
- Converted categorical variables to dummy variables for clustering.
- Standardized numeric variables for uniform scaling.
  
**2. Cluster Analysis**
- **Existing Customers**: Segmented into 4 clusters based on income, credit limit, utilization ratio, and transaction behaviors.
- Cluster 1: High-income males with unused credit limits; low utilization.
- Cluster 2: Low-income individuals with high credit utilization; financial risk.
- Cluster 3: Gold card holders with high-value transactions.
- Cluster 4: Balanced segment with moderate credit usage.

**Churned Customers**: Segmented into 3 clusters based on credit behavior and transaction patterns.
- Cluster 1: Low transaction volume; low utilization ratio.
- Cluster 2: Blue card holders with average utilization; less engagement.
- Cluster 3: High credit limit and transaction counts; unexplored attrition reasons.
  
**3. Model Validation**
- Used a hybrid clustering and decision tree model to validate cluster assignments.
- Achieved low misclassification rates:
- Existing Customers: Training (0.20%), Validation (0.24%).
- Churned Customers: Training (0.35%), Validation (0.41%).
  
**Key insights** show credit limit, transaction amounts, and income as top churn predictors. High-income customers with low utilization face less risk, while low-income and inactive segments are more prone to churn. Retention strategies include premium card upgrades, cashback rewards, and budget tools for existing customers, while re-engagement for churned customers focuses on targeted offers, low-interest cards, and real-time churn detection.

**Conclusion**
This project effectively leverages data segmentation to design actionable retention strategies. By combining clustering and decision tree models, the study achieves meaningful segmentation with minimal misclassification. Insights derived from customer profiles provide a foundation for data-driven interventions, reducing churn and enhancing customer loyalty in the banking sector.

# **Project Title: Interactive Visualization of Customer Sales Trends for Strategic Decision-Making**

**Objective:**
The goal of this project was to analyze the sales, profits, and customer behaviors of a US-based superstore in 2023. The analysis aimed to identify trends in sales performance, customer behavior, regional and segment-level patterns, and actionable insights for improving profitability and operational efficiency.

**Business Problem**
The superstore sought to:
- Understand sales trends and profits across products, customers, and regions.
- Identify underperforming products and regions.
- Optimize marketing strategies for customer retention and increased sales during key seasons.

**Skills Demonstrated**
- Data Visualization:
- Designed interactive dashboards in Tableau, allowing real-time exploration of sales metrics.
- Created comparative visualizations for year-over-year trends.
- Built region- and segment-specific analyses using charts such as bump charts, scatter plots, and treemaps.

**Problem-Solving:**
- Delivered actionable insights by analyzing customer behaviors and product-level profitability.
- Provided strategic recommendations for marketing and inventory optimization.
- 
**Storytelling with Data:**
- Highlighted insights effectively through dynamic dashboards and summary visuals, ensuring clarity for non-technical stakeholders.

**Data Visualization Process**
-**1. Sales Dashboard**
- Visualized total sales ($733K), profit ($93K), and quantity sold (12,476 units) in 2023, with a 20.36% YoY sales increase.
- Identified subcategories such as "Phones" and "Chairs" as top-performing products in terms of sales, while "Copiers" were the most profitable.
- Highlighted seasonal trends: November had the highest sales, while February was the lowest.
![image](https://github.com/user-attachments/assets/32071311-6e16-4fec-830d-17fe7e6497e6)

**2. Customer Dashboard**
- Analyzed customer distribution and purchasing behavior:
- Total customers increased by 8.6% YoY, with top customers generating 11K-14K in sales and $4K-$6K in profit.
- Highlighted the top 10 most profitable customers and their purchasing trends.
- Suggested customer retention strategies, such as loyalty programs and personalized promotions.
 ![image](https://github.com/user-attachments/assets/5e028ea3-3750-409a-9f3c-e27c1924ef91)

**3. Regional Sales Dashboard**
- Mapped sales and profits by region:
- The West region accounted for 47% of total profit, while the Central region had the lowest contribution (8%).
- Used bump charts to track regional performance over time and scatter plots to show subcategory-level performance.
![image](https://github.com/user-attachments/assets/bcb569f6-8dbe-492f-959f-8907d418f0b0)
 
**4. Segment Dashboard**
- Segmented sales by Consumer, Corporate, and Home Office categories:
- Consumers drove the majority of sales ($391K), followed by Corporate ($229K) and Home Office ($122K).
- Identified technology items like "Copiers" and "Phones" as key drivers of profitability across all segments.
![image](https://github.com/user-attachments/assets/93ec4d03-88f2-438d-8036-b19382464ad6)

**Recommendations**
**- Marketing Strategies**
- Focus on top customers during peak months (September to December) with exclusive offers and promotions.
- Increase promotional efforts in November to drive sales and convert January traffic into larger purchases.
- Implement personalized retention strategies for 1-2 order customers to increase lifetime value.
  
**Regional Strategies**
- Maximize investment in the West region while improving operations in the Central region.
- Promote high-margin products like Copiers and Phones in the East region during low-performing months.
- Segment Strategies
- Bundle office supplies with high-profit technology products to boost sales.
- Optimize inventory management for underperforming products like tables and prioritize high-demand items in peak seasons.

**Conclusion**
This project demonstrates proficiency in leveraging Tableau to extract actionable insights from large datasets. The interactive dashboards provided clarity on sales trends, regional performance, and customer behavior, enabling data-driven strategies for improving profitability and customer engagement. 
