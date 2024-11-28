#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv(r"C:\Users\Joe Ng\OneDrive\Desktop\APU\Applied_Machine_Learning\AML_Assignment\Telco-Customer-Churn.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data.dtypes


# In[8]:


data['Churn'].value_counts()


# In[9]:


data.isnull().sum()


# In[10]:


data[data.TotalCharges.isnull()]


# In[11]:


data[data['tenure'] == 0].index


# There are no additional missing values in the Tenure column.
# 
# Let's delete the rows with missing values in Tenure columns since there are only 11 rows and deleting them will not affect the data.

# In[12]:


data.drop(labels=data[data['tenure'] == 0].index, axis=0, inplace=True)
data[data['tenure'] == 0].index


# In[13]:


data.isnull().sum()


# In[14]:


#convert the total charge attribute to numeric 
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dtypes


# In[15]:


# Since the missing values is imputed with median since it's a skewed distribution  
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
data.isnull().sum()


# ## Exploratory Data Analysis EDA 

# In[16]:


import seaborn as sns 
import matplotlib.pyplot as plt 
plt.hist(data=data, x='TotalCharges')


# In[17]:


plt.figure(figsize=(3,3))
sns.countplot(x='Churn', data=data, palette='hls')
plt.show()


# In[18]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
g_labels = ['Male', 'Female']
c_labels = ['No', 'Yes']
# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=g_labels, values=data['gender'].value_counts(), name="Gender"),
              1, 1)
fig.add_trace(go.Pie(labels=c_labels, values=data['Churn'].value_counts(), name="Churn"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)

fig.update_layout(
    title_text="Gender and Churn Distributions",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                 dict(text='Churn', x=0.84, y=0.5, font_size=20, showarrow=False)])
fig.show()


# In[19]:


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Churn", y="MonthlyCharges", data=data)


# In[20]:


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Churn", y="TotalCharges", data=data)


# In[21]:


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Churn", y="TotalCharges", data=data)


# In[22]:


sns.boxplot(x="Churn", y="tenure", data=data)


# In[23]:


import plotly.express as px
fig = px.box(data, x='Churn', y = 'tenure')

# Update yaxis properties
fig.update_yaxes(title_text='Tenure (Months)', row=1, col=1)
# Update xaxis properties
fig.update_xaxes(title_text='Churn', row=1, col=1)

# Update size and title
fig.update_layout(autosize=True, width=750, height=600,
    title_font=dict(size=25, family='Courier'),
    title='<b>Tenure vs Churn</b>',
)

fig.show()


# For the people that churned, it has a lower tenure which means that it has a shorter duration with the company compared to non-churned customers that has longer tenure

# In[24]:


# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(data, hue="Churn", height=6)    .map(sns.histplot, "MonthlyCharges")    .add_legend()


# the monthly charges and tootal charges have a similar distribution among customers

# In[25]:


sns.FacetGrid(data, hue="Churn", height=5)    .map(plt.scatter, "MonthlyCharges", "TotalCharges")    .add_legend()


# In[26]:


# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop(['customerID', 'Churn'])  # Exclude non-categorical columns

# Function to plot count plots
def plot_count_plots(data, categorical_columns, target='Churn'):
    for column in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=column, hue=target, data=data)
        plt.title(f'Distribution of {column} by {target}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title=target)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Plot the count plots
plot_count_plots(data, categorical_columns)


# From here, we can see a psoitive associations of TotalCharges and MonthlyCharges among churners and non-churners

# In[27]:


data.plot(kind="scatter", x="tenure", y="TotalCharges")


# In[28]:


sns.pairplot(data.drop("customerID", axis=1), hue="Churn", height=3)


# tenure and Total Charges shows a positive association

# In[29]:


numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Function to plot histograms
def plot_histograms(data, numerical_columns):
    for column in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], bins=30, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

# Plot the histograms
plot_histograms(data, numerical_columns)


# In[30]:


sns.boxplot(data=data, x='TotalCharges')


# In[31]:


sns.boxplot(data=data, x='MonthlyCharges')


# In[32]:


sns.boxplot(data=data, x='tenure')


# In[33]:


import matplotlib.ticker as mtick
df2 = pd.melt(data, id_vars=['customerID'], value_vars=['Dependents','Partner'])
df3 = df2.groupby(['variable','value']).count().unstack()
df3 = df3*100/len(data)
colors = ['#4D3425','#E4512B']
ax = df3.loc[:,'customerID'].plot.bar(stacked=True, color=colors,
                                      figsize=(8,6),rot = 0,
                                     width = 0.2)

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers',size = 14)
ax.set_xlabel('')
ax.set_title('% Customers with dependents and partners',size = 14)
ax.legend(loc = 'center',prop={'size':14})

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',
               size = 14)


# About 50% of the customers have a partner, while only 30% of the total customers have dependents.

# In[34]:


colors = ['#4D3425','#E4512B']
partner_dependents = data.groupby(['Partner','Dependents']).size().unstack()

ax = (partner_dependents.T*100.0 / partner_dependents.T.sum()).T.plot(kind='bar',
                                                                width = 0.2,
                                                                stacked = True,
                                                                rot = 0, 
                                                                figsize = (8,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Dependents',fontsize =14)
ax.set_ylabel('% Customers',size = 14)
ax.set_title('% Customers with/without dependents based on whether they have a partner',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',
               size = 14)


# Interestingly, among the customers who have a partner, only about half of them also have a dependent, while other half do not have any independents. Additionally, as expected, among the customers who do not have any partner, a majority (90%) of them do not have any dependents .

# In[35]:



fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, sharey = True, figsize = (20,6))

ax = sns.distplot(data[data['Contract']=='Month-to-month']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'turquoise',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax1)
ax.set_ylabel('# of Customers')
ax.set_xlabel('Tenure (months)')
ax.set_title('Month to Month Contract')

ax = sns.distplot(data[data['Contract']=='One year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'steelblue',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax2)
ax.set_xlabel('Tenure (months)',size = 14)
ax.set_title('One Year Contract',size = 14)

ax = sns.distplot(data[data['Contract']=='Two year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'darkblue',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax3)

ax.set_xlabel('Tenure (months)')
ax.set_title('Two Year Contract')


# Interestingly most of the monthly contracts last for 1-2 months, while the 2 year contracts tend to last for about 70 months. This shows that the customers taking a longer contract are more loyal to the company and tend to stay with it for a longer period of time.
# 
# This is also what we saw in the earlier chart on correlation with the churn rate.

# In[36]:


#Perform Label Encoding on the Binary Values 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data['Partner'] = labelencoder.fit_transform(data['Partner'])
data['Dependents'] = labelencoder.fit_transform(data['Dependents'])
data['PhoneService'] = labelencoder.fit_transform(data['PhoneService'])
data['PaperlessBilling'] = labelencoder.fit_transform(data['PaperlessBilling'])
data.head()


# In[37]:


# Perform One-hot Encoding
data_encoded_1 = pd.get_dummies(data, columns=[
    'gender', 'OnlineBackup', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaymentMethod'
])

# Display the first few rows of the encoded dataframe
print(data_encoded_1.head())


# In[38]:


data_encoded_1['Churn'] = labelencoder.fit_transform(data_encoded_1['Churn'])
data_encoded_1.head()


# In[39]:


X = data_encoded_1.drop(columns = ['customerID','Churn'])
y = data_encoded_1['Churn'].values
# Display the shapes of X and y to confirm they are correctly separated
print(X.shape)
print(y.shape)


# # Visualize the correlation between features 

# In[40]:


# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# Display the shapes of the resulting datasets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {Y_train.shape}")
print(f"y_test shape: {Y_test.shape}")


# In[42]:


# Apply SMOTE to the training data
from imblearn.over_sampling import SMOTE
X_train_res, y_train_res = SMOTE().fit_resample(X_train, Y_train)


# In[43]:


import seaborn as sns 
sns.countplot(x = y_train_res)


# In[44]:


#Perform Min-Max Scaling 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range =(0,1)).fit(X_train_res)
X_train_norm1 = scaler.transform(X_train_res)
X_test_norm1 = scaler.transform(X_test)


# In[45]:


X_train_norm1


# In[46]:


X_test_norm1


# ## Random Forest
# 

# In[47]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
print(rf.get_params())


# In[48]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
rf.fit(X_train_norm1, y_train_res)
y_pred = rf.predict(X_test_norm1)

# Get the accuracy score
rf_acc = accuracy_score(Y_test, y_pred)*100
rf_pre = precision_score(Y_test, y_pred, average='micro')
rf_recall = recall_score(Y_test, y_pred, average='micro')
rf_f1_ = f1_score(Y_test, y_pred, average='micro')

print("\nRF - Accuracy: {:.3f}.".format(rf_acc))
print("RF - Precision: {:.3f}.".format(rf_pre))
print("RF - Recall: {:.3f}.".format(rf_recall))
print("RF - F1_Score: {:.3f}.".format(rf_f1_))
print ('\n Clasification Report:\n', classification_report(Y_test,y_pred))


# In[49]:


importances = rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')


# same as previous default parameters

# In[50]:


rf.fit(X_train_norm1, y_train_res)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# Get the predicted probabilities for the positive class
y_rfpred_prob = rf.predict_proba(X_test_norm1)[:, 1]

# Calculate the ROC curve
fpr_rf, tpr_rf, thresholds = roc_curve(Y_test, y_rfpred_prob)

# Calculate the AUC
auc_rf = roc_auc_score(Y_test, y_rfpred_prob)

# Plot the ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='SVM (AUC = {:.3f})'.format(auc_rf), color='b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve', fontsize=16)
plt.legend(loc='best')
plt.show()


# # Confusion Matrix for RF

# In[51]:


# Confusion Matrix For RF
cm = confusion_matrix(Y_test, y_pred)
print(cm)

plt.figure(figsize=(3,3))
sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r');
plt.xlabel('Predicted label');
plt.ylabel('Actual label');
plt.title("Consfusion Matrix", size = 12);


# # SVM

# In[52]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
svc= SVC(kernel='linear',probability=True)
svc.fit(X_train_norm1, y_train_res)
y_p = svc.predict(X_test_norm1)
acc=accuracy_score(Y_test, y_p)*100
print("SVM - Accuracy: {:.3f}.".format(acc))
print("\nClassification Report")
print(classification_report(Y_test, y_p))


# In[53]:


print(svc.get_params())


# In[54]:


# Train a linear SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_norm1, y_train_res)

# Extract feature importances (coefficients of the linear model)
feature_importances = svm_model.coef_[0]

# Create a DataFrame for better visualization
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Linear SVM')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Display the importance DataFrame
print(importance_df)


# # ROC Curve 

# In[55]:


# Get the predicted probabilities for the positive class
y_svcpred_prob = svc.predict_proba(X_test_norm1)[:, 1]

# Calculate the ROC curve
fpr_svc, tpr_svc, thresholds = roc_curve(Y_test, y_svcpred_prob)

# Calculate the AUC
auc_svc = roc_auc_score(Y_test, y_svcpred_prob)

# Plot the ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_svc, tpr_svc, label='SVM (AUC = {:.3f})'.format(auc_svc), color='b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve', fontsize=16)
plt.legend(loc='best')
plt.show()


# In[56]:


# Confusion Matrix For SVC
cm = confusion_matrix(Y_test, y_p)
print(cm)

plt.figure(figsize=(3,3))
sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r');
plt.xlabel('Predicted label');
plt.ylabel('Actual label');
plt.title("Consfusion Matrix", size = 12);


# # Logistic Regression

# In[57]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)
result = lr.fit(X_train_norm1, y_train_res)


# In[58]:


lr.get_params()


# In[59]:


lr.fit(X_train_norm1, y_train_res)
y_pred2 = lr.predict(X_test_norm1)

# Get the accuracy score
lr_acc = accuracy_score(Y_test, y_pred2)*100
lr_pre = precision_score(Y_test, y_pred2, average='micro')*100
lr_recall = recall_score(Y_test, y_pred2, average='micro')*100
lr_f1_ = f1_score(Y_test, y_pred2, average='micro')*100

print("\nRF - Accuracy: {:.3f}.".format(lr_acc))
print("RF - Precision: {:.3f}.".format(lr_pre))
print("RF - Recall: {:.3f}.".format(lr_recall))
print("RF - F1_Score: {:.3f}.".format(lr_f1_))
print ('\n Clasification Report:\n', classification_report(Y_test,y_pred))


# In[60]:


# Confusion Matrix For SVC
cm = confusion_matrix(Y_test, y_pred2)
print(cm)

plt.figure(figsize=(3,3))
sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r');
plt.xlabel('Predicted label');
plt.ylabel('Actual label');
plt.title("Consfusion Matrix", size = 12);


# # Plot ROC Curve

# In[61]:


# Get the predicted probabilities for the positive class
y_lrpred_prob = lr.predict_proba(X_test_norm1)[:, 1]

# Calculate the ROC curve
fpr_lr, tpr_lr, thresholds = roc_curve(Y_test, y_lrpred_prob)

# Calculate the AUC
auc_lr = roc_auc_score(Y_test, y_lrpred_prob)

# Plot the ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = {:.3f})'.format(auc_lr), color='r')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve', fontsize=16)
plt.legend(loc='best')
plt.show()


# **Feature Importance of LR**

# In[62]:


# Extract and visualize feature importance (weights)
weights = pd.Series(lr.coef_[0], index=X.columns)
plt.figure(figsize=(12, 8))
weights.sort_values(ascending=False)[:10].plot(kind='bar')
plt.title('Top 10 Features with Highest Coefficients')
plt.show()

plt.figure(figsize=(12, 8))
weights.sort_values(ascending=False)[-10:].plot(kind='bar')
plt.title('Top 10 Features with Lowest Coefficients')
plt.show()


# # Feature Selection using Principal Component Analysis (PCA)

# **Feature Selection of Logistic Regression** 

# In[64]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Standardize using Standard Scaler the data before applying PCA
scaler_std1 = StandardScaler()
X_scaled_1 = scaler_std1.fit_transform(X)

# Perform PCA
pca = PCA(n_components=0.95)  # Preserve 95% of the variance
X_pca = pca.fit_transform(X_scaled_1)

# Create a DataFrame with the PCA components
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Train a model using the PCA components and evaluate its performance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model using the PCA components
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# In[65]:


from sklearn.decomposition import PCA
# Standardize using Min max the data before applying PCA
scaler_mm = MinMaxScaler(feature_range =(0,1))
X_scaled_mm = scaler_mm.fit_transform(X)

# Perform PCA
pca = PCA(n_components=0.95)  # Preserve 95% of the variance
X_pca = pca.fit_transform(X_scaled_mm)

# Create a DataFrame with the PCA components
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Train a model using the PCA components and evaluate its performance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model using the PCA components
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# # Perform Feature Selection (PCA) 
# 

# In[66]:


# Separate features and target variable
X = data_encoded_1.drop(columns=['customerID', 'Churn'])
y = data_encoded_1['Churn'].values

# Apply SMOTE to the entire dataset (not just training data for cross-validation)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Perform Min-Max Scaling
scaler = MinMaxScaler().fit(X_res)
X_norm = scaler.transform(X_res)

# Perform PCA
pca = PCA(n_components=0.95)  # Preserve 95% of the variance
X_pca = pca.fit_transform(X_norm)

# Create a DataFrame with the PCA components
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Print the explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y_res, test_size=0.3, random_state=42)

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions and evaluate the SVM model
y_pred = svm_model.predict(X_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# # SVM Feature Selection According to Literature

# In[67]:


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Feature selection for SVM
k_svm = 17
selector_svm = SelectKBest(f_classif, k=k_svm)
X_train_svm = selector_svm.fit_transform(X_train_norm1, y_train_res)
X_test_svm = selector_svm.transform(X_test_norm1)


# In[68]:


# PCA for SVM
pca_svm = PCA(n_components=12)
X_train_svm_pca = pca_svm.fit_transform(X_train_svm)
X_test_svm_pca = pca_svm.transform(X_test_svm)


# In[69]:


# Train SVM model
svm_model = SVC(C=10, class_weight='balanced')
svm_model.fit(X_train_svm_pca, y_train_res)

# Make predictions and evaluate SVM model
y_pred_svm = svm_model.predict(X_test_svm_pca)
print("SVM Classification Report")
print(classification_report(Y_test, y_pred_svm))
print(f"SVM Accuracy: {accuracy_score(Y_test, y_pred_svm)}")


# No improvements comapred to previous model's parameters

# # Random Forest

# In[70]:


# Feature selection for Random Forest
k_rf = 16
selector_rf = SelectKBest(f_classif, k=k_rf)
X_train_rf = selector_rf.fit_transform(X_train_norm1, y_train_res)
X_test_rf = selector_rf.transform(X_test_norm1)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=33)
rf_model.fit(X_train_rf, y_train_res)

# Make predictions and evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test_rf)
print("Random Forest Classification Report")
print(classification_report(Y_test, y_pred_rf))
print(f"Random Forest Accuracy: {accuracy_score(Y_test, y_pred_rf)}")


# no improvements 

# # Apply PCA According to Odusami et al. (2021) - use this

# In[71]:


# Separate features and target variable
X = data_encoded_1.drop(columns=['customerID', 'Churn'])
y = data_encoded_1['Churn'].values

# Apply SMOTE to the entire dataset (not just training data for cross-validation)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Perform Min-Max Scaling
scaler = StandardScaler().fit(X_res)
X_norm = scaler.transform(X_res)

# Perform PCA
pca = PCA(n_components=0.95)  # Preserve 95% of the variance
X_pca = pca.fit_transform(X_norm)

# Create a DataFrame with the PCA components
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Print the explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y_res, test_size=0.3, random_state=42)

# Train Random Forest model using the transformed data
model = RandomForestClassifier(n_estimators=17, n_jobs=-1, max_depth=7, min_samples_leaf=14)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_predict2 = model.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_predict2))
print("\nClassification Report")
print(classification_report(y_test, y_predict2))
print(f"Accuracy: {accuracy_score(y_test, y_predict2)}")


# In[72]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Separate features and target variable
X = data_encoded_1.drop(columns=['customerID', 'Churn'])
y = data_encoded_1['Churn'].values

# Apply SMOTE to the entire dataset (not just training data for cross-validation)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Perform Standard Scaling
scaler = StandardScaler().fit(X_res)
X_norm = scaler.transform(X_res)

# Perform PCA
pca = PCA(n_components=0.95)  # Preserve 95% of the variance
X_pca = pca.fit_transform(X_norm)

# Create a DataFrame with the PCA components
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y_res, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model
logreg = LogisticRegression(random_state=1)
logreg.fit(X_train, y_train)

# Make predictions and evaluate the model
y_predict_lr = logreg.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_predict_lr))
print("\nClassification Report")
print(classification_report(y_test, y_predict_lr))
print(f"Accuracy: {accuracy_score(y_test, y_predict_lr)}")


# In[73]:


get_ipython().system('pip install xgboost')


# # Random Forest Hyperparameter Tuning

# **Randomized Search**

# In[75]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from numpy import arange

# Define the parameter grid with a reduced range
criterion = ['gini', 'entropy']
class_weight = ['balanced']  # 'dict' is typically used with custom class weights, keep 'balanced'
n_estimators = arange(10, 200, 20)  # Reduced range
max_features = ['auto', 'sqrt']
max_depth = arange(10, 50, 10)  # Reduced range
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the parameter grid
parameters = {
    'criterion': criterion,
    'class_weight': class_weight,
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

print(parameters)


# In[76]:


# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV
rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=parameters,
    n_iter=100,  # Number of different parameter combinations to try
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the random search model
rf_random_search.fit(X_train_norm1, y_train_res)

# Print the best parameters and best score
print("\nBest Parameters: ", rf_random_search.best_params_)
print("Best Scores: ", rf_random_search.best_score_)


# In[77]:


y_pred = rf_random_search.predict(X_test_norm1)

# Get the accuracy score
rf_acc1 = accuracy_score(Y_test, y_pred)*100
rf_pre1 = precision_score(Y_test, y_pred, average='micro')*100
rf_recall1 = recall_score(Y_test, y_pred, average='micro')*100
rf_f1_1 = f1_score(Y_test, y_pred, average='micro')*100

print("\nRF - Accuracy: {:.3f}.".format(rf_acc1))
print("RF - Precision: {:.3f}.".format(rf_pre1))
print("RF - Recall: {:.3f}.".format(rf_recall1))
print("RF - F1_Score: {:.3f}.".format(rf_f1_1))
print ('\n Clasification Report:\n', classification_report(Y_test,y_pred))
print()


# # Hyperparamter Tuning of SVC 

# **Grid Search**

# In[78]:


from sklearn.model_selection import GridSearchCV
from numpy import arange
model_svc= SVC()

parameters = dict()
parameters['kernel'] = ['rbf', 'poly', 'linear', 'sigmoid']
parameters['C'] = arange(1, 10, 1)
parameters['gamma'] = ['scale', 'auto']
parameters['class_weight'] = ['dict', 'balanced']

## Building Grid Search algorithm with cross-validation and acc score.

grid_search_svc_2 = GridSearchCV(model_svc, parameters, scoring='accuracy', cv=5, n_jobs=-1)

## Lastly, finding the best parameters.
grid_search_svc_2.fit(X_train_norm1, y_train_res)
best_parameters_SVC_2 = grid_search_svc_2.best_params_  
best_score_SVC_2 = grid_search_svc_2.best_score_ 
print()
print(best_parameters_SVC_2)
print(best_score_SVC_2)

y_pred_2 = grid_search_svc_2.predict(X_test_norm1)

# Get the accuracy score
svc_acc_2 = accuracy_score(Y_test, y_pred_2)*100
svc_pre_2 = precision_score(Y_test, y_pred_2, average='micro')
svc_recall_2 = recall_score(Y_test, y_pred_2, average='micro')
svc_f1_2 = f1_score(Y_test, y_pred_2, average='micro')

print("\nSVM - Accuracy: {:.3f}.".format(svc_acc_2))
print("SVM - Precision: {:.3f}.".format(svc_pre_2))
print("SVM - Recall: {:.3f}.".format(svc_recall_2))
print("SVM - F1 Score: {:.3f}.".format(svc_f1_2))
print ('\n Clasification Report:\n', classification_report(Y_test,y_pred_2))


# # Hyperparameter Tuning Grid Search For Logistic Regression 

# In[81]:


from sklearn.model_selection import GridSearchCV
from numpy import arange

model_LR = LogisticRegression()
print(model_LR.get_params())

parameters = dict()
parameters['random_state'] = arange(0, 100, 1) # The seed of the pseudo random number generated which is used while shuffling the data
parameters['C'] = arange(0.0001, 10, 10) # Inverse regularization parameter - A control variable that retains strength modification of Regularization by being inversely positioned to the Lambda regulator. C = 1/λ
parameters['solver'] = ['liblinear', 'newton-cg', 'lbfgs', 'saga'] # Optimization
parameters['penalty'] = ['l1', 'l2'] # Penalization (Regularization).
parameters['multi_class'] = ['auto', 'ovr', 'multinomial']

## Building Grid Search algorithm with cross-validation and acc score.
grid_search_LR = GridSearchCV(estimator=model_LR, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=-1)

## Lastly, finding the best parameters.
grid_search_LR.fit(X_train_norm1, y_train_res)
best_parameters_LR = grid_search_LR.best_params_  
best_score_LR = grid_search_LR.best_score_ 
print()
print(best_parameters_LR)
print(best_score_LR)


# In[83]:


from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
# Define the RF model
lr2 = LogisticRegression(C= 0.0001, multi_class= 'multinomial', penalty= 'l2', random_state= 0, solver= 'newton-cg')

# Define the cross-validation strategy
rkfolds_lr2 = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores_lr2 = cross_val_score(lr2, X_train_norm1, y_train_res, cv=rkfolds_lr2, scoring='accuracy')

print(accuracy_scores_lr2)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores_lr2))


# In[84]:


y_pred_1 = grid_search_LR.predict(X_test_norm1)

# Get the accuracy score
lr_acc = accuracy_score(Y_test, y_pred_1)
lr_pre = precision_score(Y_test, y_pred_1, average='micro')
lr_recall = recall_score(Y_test, y_pred_1, average='micro')
lr_f1 = f1_score(Y_test, y_pred_1, average='micro')

print("\nLR - Accuracy: {:.3f}.".format(lr_acc))
print("LR - Precision: {:.3f}.".format(lr_pre))
print("LR - Recall: {:.3f}.".format(lr_recall))
print("LR - F1 Score: {:.3f}.".format(lr_f1))
print ('\n Clasification Report:\n', classification_report(Y_test,y_pred_1))


# **Randomized Search**

# In[85]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from numpy import arange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the model
model_LR = LogisticRegression()
print(model_LR.get_params())

# Define the parameter grid
parameters = dict()
parameters['random_state'] = arange(0, 100, 1) # The seed of the pseudo random number generator
parameters['C'] = arange(0.0001, 10, 0.1) # Inverse regularization parameter
parameters['solver'] = ['liblinear', 'newton-cg', 'lbfgs', 'saga'] # Optimization
parameters['penalty'] = ['l1', 'l2', 'elasticnet', 'none'] # Penalization (Regularization)
parameters['multi_class'] = ['auto', 'ovr', 'multinomial']

# Building Randomized Search algorithm with cross-validation and accuracy score
random_search_LR = RandomizedSearchCV(estimator=model_LR, param_distributions=parameters, scoring='accuracy', cv=5, n_jobs=-1, 
                                      n_iter=100, random_state=42)

# Fit the model
random_search_LR.fit(X_train_norm1, y_train_res)
best_parameters_LR = random_search_LR.best_params_
best_score_LR = random_search_LR.best_score_

print()
print("Best Parameters: ", best_parameters_LR)
print("Best Score: ", best_score_LR)

# Make predictions
y_pred_1 = random_search_LR.predict(X_test_norm1)

# Get the accuracy score
lr_acc = accuracy_score(Y_test, y_pred_1)
lr_pre = precision_score(Y_test, y_pred_1, average='micro')
lr_recall = recall_score(Y_test, y_pred_1, average='micro')
lr_f1 = f1_score(Y_test, y_pred_1, average='micro')

print("\nLR - Accuracy: {:.3f}.".format(lr_acc))
print("LR - Precision: {:.3f}.".format(lr_pre))
print("LR - Recall: {:.3f}.".format(lr_recall))
print("LR - F1 Score: {:.3f}.".format(lr_f1))


# # Using Randomized Search on RF with PCA 

# In[86]:


# Perform PCA
pca = PCA(n_components=0.95)  # Preserve 95% of the variance
X_pca = pca.fit_transform(X_norm)

# Create a DataFrame with the PCA components
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Print the explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y_res, test_size=0.3, random_state=42)

# Define the Random Forest model
model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': arange(10, 200, 20),  # Reduced range
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Create the Randomized Search algorithm with cross-validation
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='accuracy', cv=5, n_jobs=-1,
                                   n_iter=100, random_state=42)

# Fit the model
random_search.fit(X_train, y_train)
best_parameters = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters: ", best_parameters)
print("Best Score: ", best_score)

# Train the model with the best parameters
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_predict2 = best_model.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_predict2))
print("\nClassification Report")
print(classification_report(y_test, y_predict2))
print(f"Accuracy: {accuracy_score(y_test, y_predict2)}")
print(f"Precision: {precision_score(y_test, y_predict2, average='macro')}")
print(f"Recall: {recall_score(y_test, y_predict2, average='macro')}")
print(f"F1 Score: {f1_score(y_test, y_predict2, average='macro')}")


# # Cross Validation 

# # SVM

# In[87]:


from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
# Define the SVM model
svm_model = SVC(kernel='linear', C=10, class_weight='balanced', random_state=42)

# Define the cross-validation strategy
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

# Perform cross-validation
accuracy_scores = cross_val_score(svm_model, X_train_norm1, y_train_res, cv=shuffle_split, scoring='accuracy')

print(accuracy_scores)
print("Average Accuracy after 5-Fold CV: ", np.mean(accuracy_scores))


# # 10 fold CV

# In[88]:


from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
# Define the SVM model
svm_model2 = SVC(kernel='linear', C=10, class_weight='balanced', random_state=42)

# Define the cross-validation strategy
shuffle_split1 = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

# Perform cross-validation
accuracy_scores1 = cross_val_score(svm_model2, X_train_norm1, y_train_res, cv=shuffle_split1, scoring='accuracy')

print(accuracy_scores1)
print("Average Accuracy after 10-Fold CV: ", np.mean(accuracy_scores1))


# In[89]:


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# Define the SVM model
svm_model = SVC(kernel='linear', C=10, class_weight='balanced', random_state=42)

# Define the cross-validation strategy
skfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)

# Perform cross-validation
accuracy_scores = cross_val_score(svm_model,X_train_norm1, y_train_res, cv=skfolds, scoring='accuracy')

print(accuracy_scores)
print("Average Accuracy after 10-Fold Stratified CV: ", np.mean(accuracy_scores))


# In[90]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the SVM model
svm_model = SVC(kernel='linear', C=10, class_weight='balanced', random_state=42)

# Define the cross-validation strategy
rkfolds = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores = cross_val_score(svm_model, X_train_norm1, y_train_res, cv=rkfolds, scoring='accuracy')

print(accuracy_scores)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores))


# # Using Randomized Search CV of SVC For Repeated K-fold

# In[91]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the SVM model
svm_model = SVC(kernel='rbf', C=7, gamma = 'auto', class_weight='balanced', random_state=42)

# Define the cross-validation strategy
rkfolds = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores = cross_val_score(svm_model, X_train_norm1, y_train_res, cv=rkfolds, scoring='accuracy')

print(accuracy_scores)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores))


# # SVC Cross Validation Using Grid Search 

# In[92]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the SVM model
svm_model = SVC(kernel='linear', C=6, gamma = 'scale', class_weight='balanced', random_state=42)

# Define the cross-validation strategy
rkfolds = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores = cross_val_score(svm_model, X_train_norm1, y_train_res, cv=rkfolds, scoring='accuracy')

print(accuracy_scores)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores))


# # Random Forest Cross Validation

# In[93]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the RF model
rf1 = RandomForestClassifier(n_estimators = 150, min_samples_split =10, min_samples_leaf=1, max_features='sqrt', max_depth=40,
                           criterion= 'entropy', class_weight= 'balanced', bootstrap=False)

# Define the cross-validation strategy
rkfolds1 = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores1 = cross_val_score(rf1, X_train_norm1, y_train_res, cv=rkfolds1, scoring='accuracy')

print(accuracy_scores)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores1))


# In[94]:


#Perform Shuttle Splits
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
# Define the RF model
rf_2 = RandomForestClassifier(n_estimators = 150, min_samples_split =10, min_samples_leaf=1, max_features='sqrt', max_depth=40,
                           criterion= 'entropy', class_weight= 'balanced', bootstrap=False)

# Define the cross-validation strategy
shuffle_split_rf = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

# Perform cross-validation
accuracy_scores_rf2 = cross_val_score(rf_2, X_train_norm1, y_train_res, cv=shuffle_split_rf, scoring='accuracy')

print(accuracy_scores_rf2)
print("Average Accuracy after 10-Fold CV: ", np.mean(accuracy_scores_rf2))


# In[95]:


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# Define the SVM model
rf_sk = RandomForestClassifier(n_estimators = 150, min_samples_split =10, min_samples_leaf=1, max_features='sqrt', max_depth=40,
                           criterion= 'entropy', class_weight= 'balanced', bootstrap=False)

# Define the cross-validation strategy
skfolds_rf = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)

# Perform cross-validation
accuracy_scores_rf_sk = cross_val_score(rf_sk, X_train_norm1, y_train_res, cv=skfolds_rf, scoring='accuracy')

print(accuracy_scores_rf_sk)
print("Average Accuracy after 10-Fold Stratified CV: ", np.mean(accuracy_scores_rf_sk))


# # Logistic Regression Cross-Validation

# 
# 
# 
# 
# **Repeated K Fold without parameters**

# In[96]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the RF model
lr1 = LogisticRegression()

# Define the cross-validation strategy
rkfolds_lr = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores_lr = cross_val_score(lr1, X_train_norm1, y_train_res, cv=rkfolds_lr, scoring='accuracy')

print(accuracy_scores_lr)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores_lr))


# **Repeated K-fold with parameters - It has lower accuracy scores after hyperparameter tuning** 

# In[97]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the RF model
lr_gs = LogisticRegression(C= 0.0001, multi_class= 'multinomial', penalty= 'l2', random_state= 0, solver='newton-cg')

# Define the cross-validation strategy
rkfolds_lr_gs = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores_lrgs = cross_val_score(lr_gs, X_train_norm1, y_train_res, cv=rkfolds_lr_gs, scoring='accuracy')

print(accuracy_scores_lrgs)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores_lrgs))


# In[98]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the RF model
lr_rs = LogisticRegression(solver= 'saga', random_state= 38, penalty= 'l2', multi_class= 'auto', C= 1.0001)

# Define the cross-validation strategy
rkfolds_lr_rs = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores_lr_rs = cross_val_score(lr_rs, X_train_norm1, y_train_res, cv=rkfolds_lr_rs, scoring='accuracy')

print(accuracy_scores_lr_rs)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores_lr_rs))


# **Cross_validation** 

# In[99]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the RF model
rf_pca = RandomForestClassifier(n_estimators= 150, min_samples_split= 10, min_samples_leaf= 4, 
                                max_features= 'log2', max_depth= 50, criterion= 'gini', bootstrap= True)

# Define the cross-validation strategy
rkfolds_rf_pca = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores_rfpca = cross_val_score(rf_pca, X_train_norm1, y_train_res, cv=rkfolds_rf_pca, scoring='accuracy')

print(accuracy_scores_rfpca)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores_rfpca))


# # Ensemble Learning with Grid Search

# **AdaBoost**

# In[100]:


from sklearn.ensemble import AdaBoostClassifier

ada_boost = AdaBoostClassifier(base_estimator=None)

parameters = {
    'n_estimators': [20, 50, 70, 100],
    'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'random_state' : [2, 4, 6, 8, 10]
    }

gs_ada = GridSearchCV(estimator=ada_boost, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
gs_ada.fit(X_train_norm1,y_train_res)
best_parameters = gs_ada.best_params_  
best_score = gs_ada.best_score_ 
print(best_parameters)
print(best_score)

#test acc
y_pred=gs_ada.predict(X_test_norm1)

# Get the accuracy score
ada_acc=accuracy_score(Y_test, y_pred)*100
ada_pre=precision_score(Y_test, y_pred, average='micro')*100
ada_recall=recall_score(Y_test, y_pred, average='micro')*100
ada_f1=f1_score(Y_test, y_pred, average='micro')*100

print("Adaboost - Accuracy: {:.3f}.".format(ada_acc))
print("Adaboost - Precision: {:.3f}.".format(ada_pre))
print("Adaboost - Recall: {:.3f}.".format(ada_recall))
print("Adaboost - F1_Score: {:.3f}.".format(ada_f1))


# In[101]:


from sklearn.ensemble import AdaBoostClassifier

ada_boost = AdaBoostClassifier(base_estimator=None)

parameters = {
    'n_estimators': [20, 50, 70, 100],
    'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'random_state' : [2, 4, 6, 8, 10]
    }

gs_ada = GridSearchCV(estimator=ada_boost, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1, verbose=1)
gs_ada.fit(X_train_norm1,y_train_res)
best_parameters = gs_ada.best_params_  
best_score = gs_ada.best_score_ 
print(best_parameters)
print(best_score)

#test acc
y_pred=gs_ada.predict(X_test_norm1)

# Get the accuracy score
ada_acc=accuracy_score(Y_test, y_pred)
ada_pre=precision_score(Y_test, y_pred, average='micro')
ada_recall=recall_score(Y_test, y_pred, average='micro')
ada_f1=f1_score(Y_test, y_pred, average='micro')

print("Adaboost - Accuracy: {:.3f}.".format(ada_acc))
print("Adaboost - Precision: {:.3f}.".format(ada_pre))
print("Adaboost - Recall: {:.3f}.".format(ada_recall))
print("Adaboost - F1_Score: {:.3f}.".format(ada_f1))


# **Cross-Validation of ADA Boost**

# In[102]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the RF model
ada_boost1 =AdaBoostClassifier(learning_rate =  0.3, n_estimators= 100, random_state= 2)

# Define the cross-validation strategy
rkfolds_ada = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores_ada = cross_val_score(ada_boost1, X_train_norm1, y_train_res, cv=rkfolds_ada, scoring='accuracy')

print(accuracy_scores_ada)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores_ada))


# **XgBoost Grid-Search**

# In[103]:


# pip install xgboost
from xgboost import XGBClassifier
XGB = XGBClassifier()

parameters = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
}

XGB_SE = GridSearchCV(estimator=XGB, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
XGB_SE.fit(X_train_norm1,y_train_res)
best_parameters = XGB_SE.best_params_  
best_score = XGB_SE.best_score_ 
print(best_parameters)
print(best_score)


# In[104]:





#test acc
y_pred = XGB_SE.predict(X_test_norm1)

# Get the accuracy score
XGB_acc = accuracy_score(Y_test, y_pred)*100
XGB_pre = precision_score(Y_test, y_pred, average='micro')
XGB_recall = recall_score(Y_test, y_pred, average='micro')
XGB_f1 = f1_score(Y_test, y_pred, average='micro')

print("XGB - Accuracy: {:.3f}.".format(XGB_acc))
print("XGB - Precision: {:.3f}.".format(XGB_pre))
print("XGB - Recall: {:.3f}.".format(XGB_recall))
print("XGB - F1_Score: {:.3f}.".format(XGB_f1))


# In[105]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the RF model
XGBoost =XGBClassifier(colsample_bytree= 0.6, gamma= 0.4, max_depth= 3, min_child_weight= 3, reg_alpha= 0.05, subsample= 0.92)

# Define the cross-validation strategy
rkfolds_xgb = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores_xgb = cross_val_score(XGBoost, X_train_norm1, y_train_res, cv=rkfolds_xgb, scoring='accuracy')

print(accuracy_scores_xgb)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores_xgb))


# **Extra Tree Classifier** 

# In[115]:


from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
# Define the Extra Trees Classifier model
model_ET = ExtraTreesClassifier(random_state=42)

# Define the parameter grid with a reduced range
criterion = ['gini', 'entropy']
class_weight = ['balanced']  # 'dict' is typically used with custom class weights, keep 'balanced'
n_estimators = arange(10, 200, 20)  # Reduced range
max_features = ['auto', 'sqrt']
max_depth = arange(10, 50, 10)  # Reduced range
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the parameter grid
parameters_ET = {
    'criterion': criterion,
    'class_weight': class_weight,
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

# Create the Randomized Search algorithm with cross-validation
random_search_ET = RandomizedSearchCV(estimator=model_ET, param_distributions=parameters_ET, scoring='accuracy', 
                                      cv=5, n_iter=100, random_state=42)

# Fit the model
random_search_ET.fit(X_train_norm1, y_train_res)
best_parameters_ET = random_search_ET.best_params_
best_score_ET = random_search_ET.best_score_

print("Best Parameters: ", best_parameters_ET)
print("Best Score: ", best_score_ET)

# Make predictions
y_pred_ET = random_search_ET.predict(X_test_norm1)

# Get the accuracy score
et_acc = accuracy_score(Y_test, y_pred_ET)
et_pre = precision_score(Y_test, y_pred_ET, average='micro')
et_recall = recall_score(Y_test, y_pred_ET, average='micro')
et_f1 = f1_score(Y_test, y_pred_ET, average='micro')

print("\nET - Accuracy: {:.3f}".format(et_acc))
print("ET - Precision: {:.3f}".format(et_pre))
print("ET - Recall: {:.3f}".format(et_recall))
print("ET - F1 Score: {:.3f}".format(et_f1))


# In[120]:


from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the RF model
ext_cv =ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

# Define the cross-validation strategy
rkfolds_ext = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores_ext = cross_val_score(ext_cv, X_train_norm1, y_train_res, cv=rkfolds_ext, scoring='accuracy')

print(accuracy_scores_ext)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores_ext))


# In[121]:



from sklearn.model_selection import train_test_split, RepeatedKFold
# Define the RF model
ext_cv =ExtraTreesClassifier(n_estimators= 90, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 30,
                    criterion= 'entropy', class_weight= 'balanced', bootstrap= True)

# Define the cross-validation strategy
rkfolds_ext = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)

# Perform cross-validation
accuracy_scores_ext = cross_val_score(ext_cv, X_train_norm1, y_train_res, cv=rkfolds_ext, scoring='accuracy')

print(accuracy_scores_ext)
print("Average Accuracy after Repeated 10-Fold CV: ", np.mean(accuracy_scores_ext))

