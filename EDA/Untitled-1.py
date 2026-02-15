# =================================================
# Amazon Mobiles Under 30k
# EDA + Statistical Tests
# =================================================

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------------------------------
# 1️⃣ Load Data
# -------------------------------------------------
df = pd.read_csv("amazon_mobiles_under_30k.csv")

# -------------------------------------------------
# 2️⃣ Data Cleaning
# -------------------------------------------------

# Clean Price
df["Price (INR)"] = (
    df["Price (INR)"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Clean Rating
df["Rating"] = (
    df["Rating"]
    .astype(str)
    .str.extract(r"(\d\.\d)")
    .astype(float)
)

# Drop missing values
df_clean = df.dropna(subset=["Price (INR)", "Rating"])

# -------------------------------------------------
# 3️⃣ Exploratory Data Analysis (EDA)
# -------------------------------------------------

print("\n--- DESCRIPTIVE STATISTICS ---\n")
print(df_clean[["Price (INR)", "Rating"]].describe())

# Price Distribution
plt.figure()
plt.hist(df_clean["Price (INR)"], bins=15)
plt.title("Price Distribution")
plt.xlabel("Price (INR)")
plt.ylabel("Frequency")
plt.show()

# Rating Distribution
plt.figure()
plt.hist(df_clean["Rating"], bins=10)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Price vs Rating
plt.figure()
plt.scatter(df_clean["Price (INR)"], df_clean["Rating"])
plt.title("Price vs Rating")
plt.xlabel("Price (INR)")
plt.ylabel("Rating")
plt.show()

# -------------------------------------------------
# 4️⃣ One-Sample t-test (Rating > 4.2)
# -------------------------------------------------

t_stat, p_value = stats.ttest_1samp(df_clean["Rating"], popmean=4.2)

print("\n--- ONE-SAMPLE t-TEST ---")
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value / 2 < 0.05 and t_stat > 0:
    print("Result: Reject H0 (Mean rating > 4.2)")
else:
    print("Result: Fail to reject H0")

# -------------------------------------------------
# 5️⃣ Correlation Test (Price vs Rating)
# -------------------------------------------------

corr, corr_p = stats.pearsonr(
    df_clean["Price (INR)"],
    df_clean["Rating"]
)

print("\n--- PEARSON CORRELATION ---")
print("Correlation coefficient:", corr)
print("p-value:", corr_p)

# -------------------------------------------------
# 6️⃣ ANOVA (Price difference among top brands)
# -------------------------------------------------

# Extract Brand
df_clean["Brand"] = df_clean["Product Name"].str.split().str[0]

# Select top 3 brands
top_brands = df_clean["Brand"].value_counts().head(3).index
df_anova = df_clean[df_clean["Brand"].isin(top_brands)]

groups = [
    df_anova[df_anova["Brand"] == brand]["Price (INR)"]
    for brand in top_brands
]

f_stat, anova_p = stats.f_oneway(*groups)

print("\n--- ONE-WAY ANOVA ---")
print("F-statistic:", f_stat)
print("p-value:", anova_p)

# -------------------------------------------------
# 7️⃣ Chi-Square Test (Brand vs Rating Category)
# -------------------------------------------------

# Create Rating Categories
df_clean["Rating_Category"] = pd.cut(
    df_clean["Rating"],
    bins=[0, 4.0, 4.3, 5.0],
    labels=["Low", "Medium", "High"]
)

contingency_table = pd.crosstab(
    df_clean["Brand"],
    df_clean["Rating_Category"]
)

chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n--- CHI-SQUARE TEST ---")
print("Chi-square value:", chi2)
print("Degrees of freedom:", dof)
print("p-value:", chi_p)

# -------------------------------------------------
# END OF CODE
# -------------------------------------------------
