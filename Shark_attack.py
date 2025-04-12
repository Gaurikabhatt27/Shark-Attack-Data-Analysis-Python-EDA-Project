import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.ticker import MaxNLocator

df = pd.read_csv(r"C:\Users\gauri\OneDrive\Desktop\python 4th sem\CA2\attacks.csv (1)\attacks.csv", encoding="latin1")

"⭐ Objective1: Data Cleaning and Preparation"
# Handle missing or inconsistent data, standardize date formats, and ensure categorical data is appropriately encoded.

df.columns = df.columns.str.strip().str.replace('\.', '', regex=True)

df.drop_duplicates(inplace=True)

redundant_cols = ['Case Number1', 'Case Number2', 'href formula', 'original order']
df.drop(columns=[col for col in redundant_cols if col in df.columns], inplace=True)

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Age'] = df['Age'].fillna(df['Age'].median())

df['Species'] = df['Species'].astype(str).str.strip().str.title()
df['Activity'] = df['Activity'].astype(str).str.strip().str.title()
df['Type'] = df['Type'].astype(str).str.strip().str.title()
df['Country'] = df['Country'].astype(str).str.strip().str.title()
df['Location'] = df['Location'].astype(str).str.strip().str.title()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

df = df.dropna(subset=['Case Number', 'Species'])

df.reset_index(drop=True, inplace=True)

df.to_csv('shark_attacks_cleaned.csv', index=False)

print("🔍 EXPLORATORY DATA ANALYSIS: \n")

print("✅ Shape of Cleaned Dataset:", df.shape)
print("\n🎯 Columns:\n", df.columns.tolist())
print("\n📊 Data Types:\n", df.dtypes)
print("\n📌 Missing Values:\n", df.isnull().sum())
print("\n📋 Unique Activities:\n", df['Activity'].value_counts().head(10))
print("\n🌍 Top Countries:\n", df['Country'].value_counts().head(10))

# Histogram
sns.set_style("whitegrid")
plt.style.use('dark_background')

sns.set_palette("husl")

ax = sns.histplot(df['Age'], bins=30, edgecolor='white', linewidth=0.5, alpha=0.85)
plt.title('Distribution of Ages of Shark Attack Victims', fontsize=14, pad=20)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------------------------------------------------
