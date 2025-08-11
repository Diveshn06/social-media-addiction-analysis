#!/usr/bin/env python
# coding: utf-8

# # Students Social Media Addiction — Analysis
# 
# This notebook reproduces the analysis and visualizations from the user's project. It includes data loading, cleaning, EDA, derived columns (risk level + detox strategy), and visualizations with insights.

# In[ ]:


# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 110
sns.set(style='whitegrid')


# ## 2. Load the data
# Make sure the CSV file `Students Social Media Addiction.csv` is in the same folder as this notebook.

# In[ ]:


# 2. Data Loading
data = pd.read_csv('Students Social Media Addiction.csv')
data.head()


# In[ ]:


# Data info and missing values check
print(data.shape)
print()
data.info()
print()
print("Missing values by column:")
print(data.isna().sum())


# ## 3. Optimize datatypes
# Convert suitable object columns to `category` and map Yes/No to boolean for `Affects_Academic_Performance`.

# In[ ]:


# Convert columns to category / bool where appropriate
categorical_columns = ['Gender', 'Academic_Level', 'Relationship_Status']
for col in categorical_columns:
    if col in data.columns:
        data[col] = data[col].astype('category')

if 'Affects_Academic_Performance' in data.columns:
    data['Affects_Academic_Performance'] = data['Affects_Academic_Performance'].map({'Yes': True, 'No': False})

data.info()


# ## 4. Feature engineering — Risk level and Detox strategy

# In[ ]:


# Define risk based on Addicted_Score and Mental_Health_Score
def get_risk(row):
    try:
        addicted = row['Addicted_Score']
        mental = row['Mental_Health_Score']
    except KeyError:
        return None
    if (addicted >= 8) and (mental <= 5):
        return 'High'
    elif (addicted >= 5) and (addicted < 8) and (mental <= 7):
        return 'Moderate'
    else:
        return 'Low'

def detox_strategy(risk):
    if risk == "High":
        return "Limit social media to 2 hrs/day, replace with outdoor hobbies."
    elif risk == "Moderate":
        return "Set app timers, avoid screens 1 hr before bed."
    else:
        return "Maintain current habits, focus on offline interactions."

data['Risk_Level'] = data.apply(get_risk, axis=1)
data['Detox_Strategy'] = data['Risk_Level'].apply(detox_strategy)
data[['Addicted_Score','Mental_Health_Score','Risk_Level','Detox_Strategy']].head()


# ## 5. Exploratory Data Analysis (EDA)
# Below are the visualizations reproduced from the original project.

# In[ ]:


# Distribution of Addicted_Score
plt.figure(figsize=(8,3))
sns.histplot(x=data['Addicted_Score'], bins=10, edgecolor='black', kde=True)
plt.xlabel("Addicted_Score")
plt.ylabel("Frequency")
plt.title("Distribution of Addicted_Score")
plt.grid()
plt.show()


# In[ ]:


# Age vs Avg_Daily_Usage_Hours
plt.figure(figsize=(6,3))
sns.scatterplot(x=data['Age'], y=data['Avg_Daily_Usage_Hours'])
plt.title("Age Vs Avg_Daily_Usage_Hours")
plt.grid()
plt.show()


# In[ ]:


# Gender vs Avg_Daily_Usage_Hours (boxplot)
plt.figure(figsize=(6,4))
sns.boxplot(x='Gender', y='Avg_Daily_Usage_Hours', data=data)
plt.title("Average Daily Usage by Gender")
plt.show()


# In[ ]:


# Avg Daily Usage vs Sleep hours
plt.figure(figsize=(6,4))
sns.scatterplot(x='Sleep_Hours_Per_Night', y='Avg_Daily_Usage_Hours', data=data)
plt.title("Daily Social Media Usage vs Sleep Hours")
plt.grid()
plt.show()


# In[ ]:


# Addiction score vs Academic performance
plt.figure(figsize=(6,4))
sns.boxplot(x='Affects_Academic_Performance', y='Addicted_Score', data=data)
plt.title("Addiction Score across Academic Performance")
plt.xlabel("Affects Academic Performance (True/False)")
plt.ylabel("Addicted Score")
plt.grid(True)
plt.show()


# In[ ]:


# Risk level distribution pie chart
risk_counts = data['Risk_Level'].value_counts()
labels = risk_counts.index
sizes = risk_counts.values
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor':'white'})
plt.title("Addiction Risk Level Distribution")
plt.show()


# In[ ]:


# Risk Level by Gender
plt.figure(figsize=(6,5))
sns.countplot(x='Risk_Level', hue='Gender', data=data)
plt.title("Risk Level Distribution by Gender")
plt.show()


# In[ ]:


# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(data[['Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Addicted_Score', 'Avg_Daily_Usage_Hours']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation matrix")
plt.show()


# In[ ]:


# Average addiction level by gender
gender_avg = data.groupby('Gender', observed=True)['Addicted_Score'].mean().reset_index()
gender_avg


# In[ ]:


# Average addiction level by age group
data['Age_Group'] = pd.cut(data['Age'], bins=[0,15,18,22,30], labels=['<15','15-18','19-22','23-30'])
age_avg = data.groupby('Age_Group', observed=True)['Addicted_Score'].mean().reset_index()
age_avg


# In[ ]:


# Average addiction level by education level
edu_avg = data.groupby('Academic_Level', observed=True)['Addicted_Score'].mean().reset_index()
edu_avg


# ## 6. Final Report & Insights
# - Most students fall in moderate to high addiction scores.
# - Younger students (15–22) tend to have higher scores.
# - Higher addiction correlates with reduced sleep and lower academic outcomes.
# 
# Export or run individual cells to reproduce the full analysis and plots.
