import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
file = 'menu.csv'
dataset = pd.read_csv(file)

#Display the first 5 rows
print("First 5 rows:")
print(dataset.head())

#Check for missing values
print("Missing values:")
print(dataset.isnull().sum())

#Handle missing values
print("Missing Values are droped")
dataset.dropna(inplace=True)  # Or drop missing rows

#Check the dataset types of all columns
print(dataset.dtypes)

#Summary statistics
print(dataset.describe())

# Sales by product category:'Calories'
sales_by_category = dataset.groupby('Category')['Calories'].sum().reset_index()
print("Sales by Product Category:\n", sales_by_category)

# Top items by calories assuming higher calories might indicate higher sales
top_items = dataset[['Item', 'Calories']].sort_values(by='Calories', ascending=False).head(10)
print("Top 10 Items by Calories:\n", top_items)


# Sales by product category:'Sugars' 
sales_by_category = dataset.groupby('Category')['Sugars'].sum().reset_index()
print("Sales by Product Category:\n", sales_by_category)

# Top items by Sugars 
top_items = dataset[['Item', 'Sugars']].sort_values(by='Sugars', ascending=False).head(10)
print("Top 10 Items by Sugars:\n", top_items)


# Sales by product category:'Protein'
sales_by_category = dataset.groupby('Category')['Protein'].sum().reset_index()
print("Sales by Product Category:\n", sales_by_category)

# Top items by Protein
top_items = dataset[['Item', 'Protein']].sort_values(by='Protein', ascending=False).head(10)
print("Top 10 Items by Protein:\n", top_items)


# Bar Chart: Total Calories by Product Category
plt.figure(figsize=(10, 6))
sales_by_category = dataset.groupby('Category')['Calories'].sum().reset_index()
sns.barplot(x='Category', y='Calories', data=sales_by_category)
plt.title('Total Calories by Product Category')
plt.xlabel('Category')
plt.ylabel('Total Calories')
plt.xticks(rotation=45)
plt.show()

# Line Plot: Average Calories Over Different Categories
plt.figure(figsize=(10, 6))
avg_calories_by_category = dataset.groupby('Category')['Calories'].mean().reset_index()
sns.lineplot(x='Category', y='Calories', data=avg_calories_by_category, marker='o')
plt.title('Average Calories by Product Category')
plt.xlabel('Category')
plt.ylabel('Average Calories')
plt.xticks(rotation=45)
plt.show()

# Heatmap: Correlation Matrix of Nutritional Values
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Nutritional Values')
plt.show()


