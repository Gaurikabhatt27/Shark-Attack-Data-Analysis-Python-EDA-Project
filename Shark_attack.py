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

"⭐ Objective2: Temporal Analysis"
# Analyze trends in shark attacks over time to identify patterns or anomalies.

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

sns.set_style("whitegrid")
plt.style.use('dark_background')

# Attacks per year
plt.figure(figsize=(8, 5))
sns.countplot(x='Year', data=df, hue='Year', palette='viridis', edgecolor='none')
plt.title('Number of Shark Attacks Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.xticks(rotation=90)
plt.show()

# Attacks per month
plt.figure(figsize=(10, 5))
sns.countplot(x='Month', data=df, hue='Month', palette='coolwarm', edgecolor='none')
plt.title('Number of Shark Attacks Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Attacks')
plt.show()

# Decade
df['Decade'] = (df['Year'] // 10) * 10

plt.figure(figsize=(10, 5))
sns.countplot(x='Decade', data=df, hue='Decade', palette='magma', edgecolor='white')
plt.title('Number of Shark Attacks Per Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Attacks')
plt.grid(False)
plt.show()

# Anomalies
yearly_attacks = df.groupby('Year').size()
mean_attacks = yearly_attacks.mean()
std_attacks = yearly_attacks.std()
anomalies = yearly_attacks[(yearly_attacks - mean_attacks).abs() > 2 * std_attacks]
print(anomalies)

plt.figure(figsize=(9, 5))
plt.plot(yearly_attacks.index, yearly_attacks.values, marker='x', linestyle='--', label='Attacks per Year')
plt.scatter(anomalies.index, anomalies.values, color='red', label='Anomalies')
plt.title('Shark Attacks Over Time with Anomalies Highlighted')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.legend()
plt.grid(False)
plt.show()

warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------------------------------------------------------------

"⭐ Objective3: Geospatial Analysis"
# Visualize the geographical distribution of shark attacks to identify high-risk areas.

top_countries = df['Country'].value_counts().head(3).index
df_top = df[df['Country'].isin(top_countries)]

grouped = df_top.groupby(['Country', 'Area']).size().reset_index(name='Count')
grouped = grouped.groupby('Country').apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)

plt.figure(figsize=(8, 6))
sns.barplot(y='Area', x='Count', hue='Country', data=grouped)
plt.title('Shark Attacks by Area within Top 3 Countries')
plt.xlabel('Number of Attacks')
plt.ylabel('Area')
plt.tight_layout()
plt.grid(False)
plt.show()

top_countries = df['Country'].value_counts().head(3).index
df_top = df[df['Country'].isin(top_countries)]

grouped = df_top.groupby(['Country', 'Area']).size().reset_index(name='Count')
low_risk_grouped = grouped.groupby('Country').apply(lambda x: x.nsmallest(5, 'Count')).reset_index(drop=True)

low_risk_grouped['Label'] = low_risk_grouped['Country'] + ' | ' + low_risk_grouped['Area']
low_risk_grouped = low_risk_grouped.sort_values(by='Count', ascending=True)

plt.figure(figsize=(10, 8))
colors = sns.color_palette("Set2", n_colors=len(top_countries))
country_color = dict(zip(top_countries, colors))

for _, row in low_risk_grouped.iterrows():
    plt.hlines(y=row['Label'], xmin=0, xmax=row['Count'], color=country_color[row['Country']], linewidth=2)
    plt.plot(row['Count'], row['Label'], "o", markersize=7, color=country_color[row['Country']])

for country in top_countries:
    plt.plot([], [], color=country_color[country], label=country)
plt.legend(title='Country')

plt.title('Low-Risk Shark Attack Areas (Top 3 Countries)', fontsize=14)
plt.xlabel('Number of Attacks')
plt.ylabel('Area')
plt.tight_layout()
plt.legend(title='Country', loc='lower right', fontsize=12)
plt.grid(False)
plt.show()
# -----------------------------------------------------------------------------------------------------------------------------

"⭐ Objective4: Yearly and Global Trend Analysis"
# Analyzing how shark attack incidents have evolved over time and identifying countries most affected globally.

plt.style.use('dark_background')
# sns.set_style("darkgrid")

df_copy = df.copy()  

df_filtered = df_copy[['Year', 'Country']]

df_filtered = df_filtered[df_filtered['Year'].apply(lambda x: str(x).isdigit())]
df_filtered['Year'] = df_filtered['Year'].astype(int)

df_filtered = df_filtered[df_filtered['Year'] > 1900]

df_filtered = df_filtered.dropna(subset=['Country'])

attacks_per_year = df_filtered['Year'].value_counts().sort_index()
cumulative_attacks = attacks_per_year.cumsum()

top_countries = df_filtered['Country'].value_counts().head(10)

plt.style.use('dark_background')
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
plt.subplots_adjust(hspace=0.4)
fig.subplots_adjust(bottom=0.1)

# Yearly trend plot
axs[0].plot(attacks_per_year.index, attacks_per_year.values, marker='x', color='hotpink')
axs[0].set_title('Yearly Trend of Shark Attacks')
axs[0].set_xlabel('Year', labelpad=10)
axs[0].set_ylabel('Number of Attacks')
axs[0].grid(False)
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
axs[0].tick_params(axis='x', rotation=45)

# Cumulative attacks plot
axs[1].fill_between(cumulative_attacks.index, cumulative_attacks.values, color='#7A3B7B', alpha=0.5)
axs[1].plot(cumulative_attacks.index, cumulative_attacks.values, color='#9E5D85')
axs[1].set_title('Cumulative Shark Attacks Over Time')
axs[1].set_xlabel('Year', labelpad=10)
axs[1].set_ylabel('Cumulative Attacks')
axs[1].grid(False)
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
axs[1].tick_params(axis='x', rotation=45)

# Top countries plot
axs[2].barh(top_countries.index[::-1], top_countries.values[::-1], color='#D27E9C', edgecolor="#D27E9C")
axs[2].set_title('Top 10 Countries with Shark Attacks')
axs[2].set_xlabel('Number of Attacks')
axs[2].set_ylabel('Country')
axs[2].grid(False)

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------

"⭐ Objective5: Fatality Distribution and Impact"
# Analyzing the severity and demographic impact of fatal vs non-fatal incidents.

df.columns = df.columns.str.strip()  
df.rename(columns={'Fatal (Y/N)': 'Fatal'}, inplace=True)
df['Fatal'] = df['Fatal'].str.upper().str.strip()

df['Fatal'] = df['Fatal'].replace({'UNKNOWN': 'N', '2017': 'N'})  # Replacing with 'N' (Non-Fatal)

df = df[df['Fatal'].isin(['Y', 'N'])]

# --- 1. Donut Chart: Fatal vs Non-Fatal ---
fatal_counts = df['Fatal'].value_counts()

# Plot the donut chart
plt.figure(figsize=(6, 6))
colors=['#8B4513', '#D2B48C']
plt.pie(fatal_counts, labels=fatal_counts.index, autopct='%1.1f%%', startangle=90,
        colors=colors, wedgeprops=dict(width=0.4, edgecolor='none'))
plt.title('Fatal vs Non-Fatal Attacks')
plt.show()

# --- 2. Violin Plot: Age and Fatality Distribution by Gender ---
df_violin = df[['Age', 'Sex', 'Fatal']].dropna()
df_violin['Age'] = pd.to_numeric(df_violin['Age'], errors='coerce')
df_violin = df_violin.dropna()

# Plot violin plot for Age and Fatality by Gender
plt.figure(figsize=(8, 6))
sns.violinplot(x='Sex', y='Age', hue='Fatal', data=df_violin, split=True,
               palette={'Y': '#ff6666', 'N': '#66b3ff'})
plt.title('Age and Fatality Distribution by Gender')
plt.show()
#------------------------------------------------------------------------------------------------------------------------

"⭐ Objective6: Demographic and Fatality Analysis"
# Analyze the relationship between age, year, and gender with the fatality status of shark attacks to identify demographic patterns influencing attack outcomes.

df['Fatal'] = df['Fatal'].map({'Y': 1, 'N': 0})

subset_df = df[['Age', 'Year', 'Fatal']].dropna()

correlation_matrix = subset_df.corr()
correlation_matrix_copy = correlation_matrix.copy()

#Heat Map
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, square=True)
plt.title('Correlation Heatmap of Shark Attack Dataset')
plt.tight_layout()
plt.show()

# Scatter Plot

df['Sex'] = df['Sex'].replace(['N', 'lli'], np.nan)  

df = df.dropna(subset=['Sex'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Year', hue='Sex', palette='husl', style='Sex', markers={'M': 'o', 'F': 'X'}, edgecolor='none')
plt.title('Scatter Plot of Age vs Year of Shark Attack')
plt.xlabel('Age')
plt.ylabel('Year')
plt.tight_layout()
plt.grid(False)
plt.show()
#---------------------------------------------------------------------------------------------------------------------------

"⭐ Objective7: Outlier Detection in Numerical Data"
# Identifying anomalies in columns like Age and Year to improve data accuracy and analytical insights.

numeric_cols = df.select_dtypes(include='number').drop(['Year', 'Decade'], axis=1)

# Plot boxplots for each numeric column
plt.figure(figsize=(12, 6))
sns.boxplot(data=numeric_cols, palette='husl')
plt.title('Boxplots of Numerical Columns (excluding Year and Decade)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

year_decade_df = df[['Year', 'Decade']]

# Plot combined boxplot for 'Year' and 'Decade'
plt.figure(figsize=(8, 6))
sns.boxplot(data=year_decade_df, palette='muted')
plt.title('Boxplot of Year and Decade')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

