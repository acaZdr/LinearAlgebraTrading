import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('../results/outputs/times.csv')

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Ensure numeric columns (convert if necessary)
df['Test Loss'] = pd.to_numeric(df['Test Loss'], errors='coerce')
df['Average Dollar Difference ($)'] = pd.to_numeric(df['Average Dollar Difference ($)'], errors='coerce')

# Drop rows with missing values if any
df = df.dropna()

# Plot Test Loss by PCA vs Non-PCA
plt.figure(figsize=(12, 6))
sns.boxplot(x='PCA', y='Test Loss', data=df, palette='coolwarm')
plt.title('Test Loss by PCA vs Non-PCA')
plt.xlabel('PCA')
plt.ylabel('Test Loss')
plt.grid(True)
plt.show()

# Plot Average Dollar Difference by PCA vs Non-PCA
plt.figure(figsize=(12, 6))
sns.boxplot(x='PCA', y='Average Dollar Difference ($)', data=df, palette='coolwarm')
plt.title('Average Dollar Difference by PCA vs Non-PCA')
plt.xlabel('PCA')
plt.ylabel('Average Dollar Difference')
plt.grid(True)
plt.show()

# Plot Test Loss Over Data Limit by PCA vs Non-PCA
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Data Limit', y='Test Loss', hue='PCA', palette='coolwarm', marker='o')
plt.title('Test Loss Over Data Limit by PCA vs Non-PCA')
plt.xlabel('Data Limit')
plt.ylabel('Test Loss')
plt.legend(title='PCA')
plt.grid(True)
plt.show()

# Plot Average Dollar Difference Over Data Limit by PCA vs Non-PCA
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Data Limit', y='Average Dollar Difference ($)', hue='PCA', palette='coolwarm', marker='o')
plt.title('Average Dollar Difference Over Data Limit by PCA vs Non-PCA')
plt.xlabel('Data Limit')
plt.ylabel('Average Dollar Difference')
plt.legend(title='PCA')
plt.grid(True)
plt.show()
