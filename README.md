##  **Prediction of Customer Churn Using Machine Learning and Ensemble Techniques**
This project aims to predict customer churn in the telecommunications industry using various machine learning models, including Logistic Regression, Random Forest, and Support Vector Machine, alongside ensemble methods like XGBoost, AdaBoost, and Extra Trees Classifier. Key data preprocessing techniques such as SMOTE for class balancing and Principal Component Analysis (PCA) for dimensionality reduction were utilized. The Random Forest model optimized with Randomized Search CV achieved the highest accuracy of 86.10%, followed closely by XGBoost at 85.84%. The insights derived from this analysis support proactive customer retention strategies.

**Skills Demonstrated**
- Machine Learning Models: Logistic Regression, Random Forest, Support Vector Machine (SVM), AdaBoost, XGBoost, Extra Trees Classifier.
- Data Preprocessing: Feature engineering, handling missing values, label encoding, one-hot encoding, and class balancing with SMOTE.
- Dimensionality Reduction: PCA with variance preservation (95%).
- Hyperparameter Tuning: Grid Search and Randomized Search CV for model optimization.
- Model Validation: 10-fold and repeated K-fold cross-validation to assess model robustness.
- Evaluation Metrics: ROC-AUC, F1-score, precision, recall, and confusion matrices.
- Programming Libraries: NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, Plotly, Imbalanced-learn.

**Process Breakdown**
**1. Dataset Overview**
- Source: Kaggle’s Telco Customer Churn Dataset (7043 instances, 21 features). 
- Features: Customer demographics, account details, and payment history (e.g., gender, tenure, total charges, monthly charges, contract type). 
- Target Variable: Churn (binary classification: churned vs. non-churned).

**2. Data Preprocessing**

- Class Balancing: Addressed 26.6% churn rate imbalance using **SMOTE**.

- Feature Encoding: Converted binary and categorical features using**label** and **one-hot encoding**.

- Scaling: **Min-Max Scaling** applied to normalize skewed numerical data.
  

**3. Exploratory Data Analysis (EDA)**
Insights:
- Higher monthly charges and shorter tenure correlate with increased churn.
- Non-churners are predominantly on two-year contracts with lower monthly charges.
- Churners commonly use electronic checks as their payment method.
- Visualizations: **Correlation heatmaps, boxplots, and distribution charts highlighted key trends**.
  
![image](https://github.com/user-attachments/assets/242ddf34-eb51-483f-9b46-74465887876b)

![image](https://github.com/user-attachments/assets/c2845634-3209-4cbd-955d-a6137fe5fe5e)

![image](https://github.com/user-attachments/assets/f5d27d8e-b06b-4aae-b804-7058243d4a8c)

![image](https://github.com/user-attachments/assets/ba3e996d-5100-4de4-9151-17beb066b0a2)

![image](https://github.com/user-attachments/assets/da4b12fc-2f06-4701-be32-e204db0316c7)


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
  
**Performance:**
- RF with Randomized Search Hyperparameters (n_estimators= 150, min_samples_split= 10, min_samples_leaf= 4, max_features= 'log2', max_depth= 50, criterion= 'gini', bootstrap= True) achieved the highest accuracy of **86.10%**
  
![image](https://github.com/user-attachments/assets/8024f8f6-e25b-498e-bd6d-a86dd973ef92)

**Comparison of the ensemble models** 
- Best Performance: **XGBoost** demonstrated the **highest accuracy (85.84%)** and adaptability with hyperparameter tuning and 10 fold cross-validation. 
- Default Performance: Achieved an accuracy of 78.39%, with precision, recall, and F1 scores at par with accuracy.
- **Tuned Performance:** Best parameters included max_depth=3, colsample_bytree=0.6, and gamma=0.4.


**Conclusion**: 
This project demonstrates the application of advanced machine learning techniques and model optimization strategies to solve a real-world business problem. The resulting models provide actionable insights for targeted customer retention efforts, enabling telecommunications firms to minimize churn and maximize customer lifetime value.

# **Project Title: Hotel Reservation Cancellation Prediction Using Data Analysis and Statistical Techniques in SAS**

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
- Handling Missing Values

- Replaced categorical missing values with the mode (e.g., "Online" for market segment).

- Removed rows with missing reservation dates due to their minimal impact (<0.02%).
 
**Outlier Treatment:**
- Identified and removed outliers in numerical variables like average price using IQR thresholds.


**Feature Engineering:**
- Feature Creation & One-hot Encoding

**Visualization:**
- Boxplots and histograms revealed that canceled bookings were skewed toward higher lead times and prices.
- Bar charts highlighted cancellation trends across price categories and market segments. new variables such as:
  a. Total Nights: Sum of weekend and weekday nights.
  b. Total Customers: Sum of adults and children per booking.
  c. ADR (Average Daily Rate): Average price divided by total nights.
  
**Key Insights:**
- Most bookings involved 2 adults and lasted 2-3 nights.
- Lead Time: Canceled bookings had a mean lead time of 137 days, while non-canceled bookings averaged 59 days.
- Average Price: Higher-priced bookings (> $100/day) had a significantly higher cancellation rate (40% vs. 24.5% for lower prices).
- Market Segment: Online bookings showed the highest cancellation rate (37.56%), while offline and complementary bookings had lower rates.


**Examples of EDA:**

![image](https://github.com/user-attachments/assets/c1a316b1-cafc-4690-b55f-edc9babcc9e7)

![image](https://github.com/user-attachments/assets/d029a297-ddc8-498d-a91f-a61488fe33c9)

![image](https://github.com/user-attachments/assets/070acf74-8ca4-437f-b45e-16a8c2fe23c0)

![image](https://github.com/user-attachments/assets/8a59f76d-3da4-4314-9c44-4db057934c5e)

**Statistical Analysis**
- **T-Test**: Lead times for canceled bookings were significantly longer than non-canceled bookings (mean 137.5 days vs. 59.05 days, p < 0.0001).

- **ANOVA**: Higher room prices and certain room types were significantly associated with cancellations (p < 0.0001). Booking status explained 2.11% of price variability.

- **Correlation**: Moderate positive correlation between lead time and cancellations (R = 0.43). Weak correlation between special requests and booking status.

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
- SAS Enterprise Miner (Pipeline)
  
  ![image](https://github.com/user-attachments/assets/c94a94a2-3931-42e8-bcad-70602edc4ff6)

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

**Existing Customer Cluster Profile**

  ![image](https://github.com/user-attachments/assets/54f18faa-23fd-4faa-9476-8b165794ff4e)

**Churned Customers**: Segmented into 3 clusters based on credit behavior and transaction patterns.
- Cluster 1: Low transaction volume; low utilization ratio.
- Cluster 2: Blue card holders with average utilization; less engagement.
- Cluster 3: High credit limit and transaction counts; unexplored attrition reasons.
 
**Attrited Customer Cluster Profile**

  ![image](https://github.com/user-attachments/assets/b903ee55-a070-4393-8beb-347685755a59)

  
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
  
**Segment Strategies**
- Bundle office supplies with high-profit technology products to boost sales.
- Optimize inventory management for underperforming products like tables and prioritize high-demand items in peak seasons.

**Conclusion**
This project demonstrates proficiency in leveraging Tableau to extract actionable insights from large datasets. The interactive dashboards provided clarity on sales trends, regional performance, and customer behavior, enabling data-driven strategies for improving profitability and customer engagement. 

# **Loan Approval Prediction using SQL and SAS**
**Project Background**
LFI, headquartered in New York, provides financial support to SMEs. The manual loan approval process faced inefficiencies due to high application volumes, human errors, and time constraints. Automating this process was crucial to streamline decision-making, minimize risks, and enhance business credibility.

**Problem Statement**
Loan approval requires evaluating applicants’ financial profiles, including income, credit history, and loan amount. Manual evaluations led to inconsistencies and missed opportunities. This project aimed to leverage historical data for developing a predictive model to classify loan applicants into high or low risk, enabling efficient and fair loan approvals.

**Literature Review**
- Loan Approval Factors: Studies (Haron et al., 2013) emphasized collateral, financial stability, and credit history as key determinants for loan approval. The "5Cs" framework (Character, Capacity, Capital, Collateral, and Conditions) further standardized evaluations.
- Machine Learning Models: Logistic regression and Random Forest models emerged as effective predictors of loan defaults (Serengil et al., 2021). Random Forest achieved 92% accuracy, outperforming traditional methods.

  
**Skills Demonstrated**

**1. Data Preparation and Cleaning**
- Dataset: The data comprised 614 training observations with variables such as gender, marital status, income, loan amount, and approval status.
  
**Techniques Used:**
- **Identified and addressed missing values:**
- **Categorical Variables**: Imputed missing values using the mode (e.g., gender).

  ![image](https://github.com/user-attachments/assets/840738bc-54bf-417f-b4a6-5b391c7dcb8f)

- **Continuous Variables**: Imputed missing values using the mean or median based on skewness.

  ![image](https://github.com/user-attachments/assets/fd2f6dab-a152-44df-9b8c-60c2df08e012)

- **Outlier detection** ensured data consistency for continuous variables like income and loan amount.
  
**2. Exploratory Data Analysis (EDA)**
- Conducted univariate analysis to examine distributions:
- Loan Amount: Positively skewed with most loans between $100K–$200K.
- Candidate Income: Highlighted significant income disparities among applicants.
- Performed bivariate analysis to uncover relationships:
- Married individuals had a higher loan approval rate (68%) compared to unmarried.
- Employed candidates received higher loan amounts on average.

**Examples of EDA:** 

- Univariate Analysis:
  
  ![image](https://github.com/user-attachments/assets/743e5398-8dd5-40a9-bd49-73c67dbcc370)

  ![image](https://github.com/user-attachments/assets/d0fcbcd3-a2b9-4ce6-b2d1-01c7855047ac)

- Bivariate Analysis:

  ![image](https://github.com/user-attachments/assets/33bacd5d-807d-4ad1-87fe-5077753d6fa3)

  ![image](https://github.com/user-attachments/assets/66671058-4862-4623-bdda-6477bac492be)

**3. Predictive Modeling**
- Developed a Logistic Regression model in SAS to predict loan approval status:
- Dependent Variable: Loan approval (Yes/No).
- Independent Variables: Income, employment status, loan amount, and credit history.
- Achieved model interpretability by analyzing feature significance.

  ![image](https://github.com/user-attachments/assets/0d87a352-3847-451d-a23e-392eae65a6da)


**4. SAS Macro Expertise**
- Utilized SAS Macros to automate querying and analysis by enhancing efficiency and reproducibility.

  ![image](https://github.com/user-attachments/assets/911015af-29aa-4cf3-930f-0d9ed94d923f)
  
  ![image](https://github.com/user-attachments/assets/60cba531-b76f-4093-8636-c34374a7ee13)

**Key Achievements**
- Data Imputation: Addressed 15% of missing values across variables, improving model reliability.
- Model Accuracy: The logistic regression model achieved 85% predictive accuracy, enabling actionable decisions.
  
**Data Insights:**
- Males constituted 80% of applicants, highlighting a gender disparity in financial service access.
- Majority of loans originated from urban areas, indicating location-based access differences.
  
**Conclusion**
This project demonstrated my proficiency in data analysis, modeling, and using tools like SAS and SQL to solve complex problems. By automating loan approvals, I contributed to LFI’s operational efficiency and decision-making capabilities. These skills reflect my ability to handle data-driven challenges and deliver impactful solutions.

# **Customer Retention Strategies Using SAP Lumira Discovery**

**Introduction**
I utilized SAP Lumira Discovery to analyze and visualize customer retention strategies for Global Bike Inc. (GBI). Through comprehensive data exploration and insights generation, I developed actionable strategies to enhance customer retention and revenue. My work demonstrates expertise in data analysis, business intelligence, and visualization tools, tailored to address organizational goals.

**Objective**
The primary aim of this project was to:
- Analyze customer churn and retention trends in the USA from 2016 to 2022.
- Propose data-driven strategies for customer retention to boost revenue using insights from SAP Lumira.

**Data Preparation**
The dataset, extracted from GBI's ERP system, spanned sales data from 2016 to 2023 (35030 rows, 51 attributes). 
Key variables included:
- Temporal attributes (Year, Quarter, Month)
- Customer attributes (City, State, Customer Name)
- Product metrics (Quantity, Revenue, Price, Profit Margin, Discount) Using SAP Lumira Discovery, I transformed raw data into visual dashboards, emphasizing customer segmentation, sales performance, and product profitability.
  
**Analysis and Insights**

**Revenue Trends**
- Identified seasonal revenue peaks in Q2 and Q4, with sharp declines in Q3.
- High-revenue customers (e.g., Rocky Mountain Bike, Socal Bikes) contributed the majority of profits, while low-revenue customers showed churn patterns.

  ![image](https://github.com/user-attachments/assets/b0b2f6c0-b1f0-4003-97ab-4d4a568e0c81)

  ![image](https://github.com/user-attachments/assets/73c09d10-f899-4695-933f-49cc94b535aa)


**Product Performance**
- Professional Touring Bikes (various colors) consistently generated the highest profit margins.
- Men’s and Women’s Off-Road Bikes had lower sales and profit contributions, indicating opportunities for strategic bundling or marketing.

  ![image](https://github.com/user-attachments/assets/ba8b1dfc-a2ce-4ec7-90c7-64a33df4d9f1)

**Discount Strategies**
- Significant discounts on top-performing products (e.g., Professional Touring Bikes) boosted revenue.
- Lack of discounts for some customers correlated with churn (e.g., DC Bikes).
  
**Regional Analysis**
- US West customers contributed higher revenue and profit margins compared to the US East.
- Accessories, such as elbow pads and knee pads, showed stable demand but lower profit margins.
  
  ![image](https://github.com/user-attachments/assets/0a835033-ae72-4213-90c5-472e044e41d7)

  ![image](https://github.com/user-attachments/assets/85445974-640d-4b20-a89a-ace82d7381e4)

**Outcomes and Deliverables**

a. Balanced Discount Strategies
- Periodic discounts on high-demand products to retain top customers.
- Focused discounts for at-risk customers to prevent churn.

b. Seasonal Marketing Campaigns
- Targeted promotions during peak revenue periods (Q2, Q4).
- Incentivized sales during low-demand quarters (Q3).
  
c. Loyalty Programs
- Reward systems for high-value customers to encourage repeat purchases.
  
d. Product Optimization
- Focused on high-revenue products (e.g., Touring Bikes).
- Bundled underperforming products with popular items to increase sales.

**Conclusion**
This project highlighted my ability to utilize SAP Lumira Discovery for advanced data analysis and visualization. By applying a structured approach to customer retention, I developed evidence-based strategies that align with organizational goals. My expertise in business intelligence tools and data-driven decision-making equips me to deliver impactful results in dynamic business environments.
