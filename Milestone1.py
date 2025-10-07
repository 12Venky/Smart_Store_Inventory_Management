import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Venkateshwara Rao\Codes\retail_store_inventory.csv")

print("Data Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

fill_cols = ['Units Sold', 'Inventory Level', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing']
df[fill_cols] = df[fill_cols].fillna(0)

df = df.drop_duplicates()
df = df.sort_values(by=['Product ID', 'Date'])

if 'Holiday/Promotion' in df.columns:
    df['Holiday/Promotion'] = df['Holiday/Promotion'].fillna(0)

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered

for col in ['Units Sold', 'Inventory Level', 'Units Ordered', 'Price', 'Discount']:
    df = remove_outliers_iqr(df, col)

df['units_sold_ma7'] = df.groupby('Product ID')['Units Sold'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['lag_1'] = df.groupby('Product ID')['Units Sold'].shift(1)
df['lag_7'] = df.groupby('Product ID')['Units Sold'].shift(7)
df['lag_14'] = df.groupby('Product ID')['Units Sold'].shift(14)
df['price_diff'] = df['Price'] - df['Competitor Pricing']
df['revenue'] = df['Units Sold'] * df['Price']
df['discount_flag'] = (df['Discount'] > 0).astype(int)
df['day_of_week'] = df['Date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['week'] = df['Date'].dt.isocalendar().week
df['month'] = df['Date'].dt.to_period('M')

products = df['Product ID'].unique()
for p in products[:3]:
    temp = df[df['Product ID'] == p]
    plt.figure(figsize=(10,4))
    plt.plot(temp['Date'], temp['Units Sold'], label="Daily Units Sold")
    plt.plot(temp['Date'], temp['units_sold_ma7'], label="7-Day MA")
    plt.title(f"Units Sold Trend - {p}")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.show()

category_sales = df.groupby(['Category', 'Date'])['Units Sold'].sum().reset_index()
categories = category_sales['Category'].unique()

for c in categories[:3]:
    temp = category_sales[category_sales['Category'] == c]
    plt.figure(figsize=(10,4))
    plt.plot(temp['Date'], temp['Units Sold'])
    plt.title(f"Units Sold Trend by Category - {c}")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Units Sold'], bins=30, kde=True)
plt.title("Units Sold Distribution")
plt.show()

monthly_sales = df.groupby(['month'])['Units Sold'].sum().reset_index()
plt.figure(figsize=(10,4))
plt.plot(monthly_sales['month'].astype(str), monthly_sales['Units Sold'])
plt.title("Monthly Units Sold Trend")
plt.xticks(rotation=45)
plt.show()

numeric_cols = ['Units Sold', 'Inventory Level', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing', 'price_diff', 'revenue']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()

df.to_csv("cleaned_retail_data.csv", index=False)
print("\nPreprocessed data with outlier handling & feature engineering saved as cleaned_retail_data.csv") 