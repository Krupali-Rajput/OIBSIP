import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
file = 'customer_shopping_data.csv'
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

#mode
mode_price = dataset['price'].mode()[0]
print(f"Mode price: {mode_price}")

mode_age = dataset['age'].mode()[0]
print(f"Mode age: {mode_age}")

mode_quantity = dataset['quantity'].mode()[0]
print(f"Mode quantity: {mode_quantity}")

#Group sales by price
sales_by_price = dataset.groupby('price')['quantity'].sum()


#Plot sales over time
plt.figure(figsize=(10, 6))
sales_by_price.plot(title='Sales Over Time')
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.show()

#Optional: Moving average
sales_by_price.rolling(window=7).mean().plot(title='7-Day Moving Average of Sales')
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.show()

#Grouping customers by demographic features
customer_demographics = dataset.groupby(['gender', 'age'])['customer_id'].nunique().reset_index()

#Rename columns
customer_demographics.columns = ['gender', 'age', 'Unique_Customers']

#Display the demographics data
print("Customer Demographics Analysis:")
print(customer_demographics.head())

#Analyze purchasing behavior by demographics
if 'price' in dataset.columns:
    purchasing_behavior = dataset.groupby(['gender', 'age'])['price'].mean().reset_index()
    purchasing_behavior.columns = ['gender', 'age', 'Average_Sales']
    
    print("Purchasing Behavior by Demographics:")
    print(purchasing_behavior.head())

#Analyze sales by product category
if 'category' in dataset.columns:
    sales_by_category = dataset.groupby('category')['price'].sum().reset_index()
    sales_by_category.columns = ['category', 'Total_Price']

    print("Sales by Product Category:")
    print(sales_by_category.head())

#Analyze the frequency of purchases by product
if 'category' in dataset.columns:
    purchase_frequency_by_category = dataset.groupby('category')['customer_id'].count().reset_index()
    purchase_frequency_by_category.columns = ['category', 'Purchase_Frequency']
    
    print("Purchase Frequency by Product Category:")
    print(purchase_frequency_by_category.head())

#Top customers based on total spending
if 'customer_id' in dataset.columns and 'price' in dataset.columns:
    top_customers = dataset.groupby('customer_id')['price'].sum().reset_index()
    top_customers = top_customers.sort_values(by='price', ascending=False).head(10)
    
    print("Top 10 Customers by Total Spending:")
    print(top_customers)


if 'category' in dataset.columns and 'price' in dataset.columns:
    sales_by_category = dataset.groupby('category')['price'].sum().reset_index()

    #Bar chart for Sales by Product Category
    plt.figure(figsize=(10, 6))
    sns.barplot(x='category', y='price', data=sales_by_category)
    plt.title('Total Sales by Product Category')
    plt.xlabel('Product Category')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.show()


if 'quantity' in dataset.columns and 'price' in dataset.columns:
    sales_by_date = dataset.groupby('quantity')['price'].sum().reset_index()

    #Line plot for Sales Over Time
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='quantity', y='price', data=sales_by_date)
    plt.title('Sales Over Time')
    plt.xlabel('quantity')
    plt.ylabel('price')
    plt.xticks(rotation=45)
    plt.show()

#Heatmap for correlation matrix if there are numerical columns
plt.figure(figsize=(10, 6))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



