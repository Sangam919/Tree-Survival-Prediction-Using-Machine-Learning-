import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("clean_tree_data1.csv")

# BASIC INFO
print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nTarget Distribution:\n", df["Target"].value_counts())

# 🔧 FIXES (IMPORTANT)

# Ensure Target is numeric (0/1)
df["Target"] = pd.to_numeric(df["Target"], errors="coerce")

# Numerical columns
numerical_cols = [
    "Avg_Temperature_C",
    "Rainfall_mm",
    "Planted_Trees",
    "Water_Frequency_per_Week"
]

# Convert numerical columns properly
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values (for clean plots)
df = df.dropna(subset=numerical_cols + ["Target"])

# TARGET DISTRIBUTION (BAR)

plt.figure(figsize=(6,4))
sns.countplot(x="Target", data=df)
plt.title("Target Variable Distribution")
plt.xlabel("Survival (0 = Low, 1 = High)")
plt.ylabel("Count")
plt.show()

# PIE CHART
plt.figure(figsize=(5,5))
df["Target"].value_counts().plot.pie(
    autopct="%1.1f%%",
    labels=["Low Survival", "High Survival"],
    startangle=90
)
plt.title("Survival Percentage")
plt.ylabel("")
plt.show()

#  HISTOGRAMS

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# BOX PLOTS 

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Target", y=col, data=df)
    plt.title(f"{col} vs Survival")
    plt.xlabel("Survival (0 = Low, 1 = High)")
    plt.ylabel(col)
    plt.show()

# CORRELATION HEATMAP
plt.figure(figsize=(10,8))
corr = df[numerical_cols + ["Target"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# FEATURE vs TARGET (BAR)
plt.figure(figsize=(6,4))
sns.barplot(x="Soil_Quality", y="Target", data=df)
plt.title("Soil Quality vs Survival Rate")
plt.xlabel("Soil Quality")
plt.ylabel("Survival Rate")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x="Area_Type", y="Target", data=df)
plt.title("Area Type vs Survival Rate")
plt.xlabel("Area Type")
plt.ylabel("Survival Rate")
plt.show()
