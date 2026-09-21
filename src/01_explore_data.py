# Pandas library import kar rahe hain, "pd" chhota naam hai taaki baar-baar poora na likhna pade
import pandas as pd
# Path check karne ke liye (file hai ya nahi)
from pathlib import Path

# Dataset ka path ek variable mein rakha.
# Capital letters = constant (ye value program mein change nahi hogi)
DATA_PATH = "data/raw/telco_churn.csv"

# Agar file nahi mili toh confusing error ki jagah saaf message dikhao
if not Path(DATA_PATH).exists():
    raise SystemExit(
        f"File nahi mili: {DATA_PATH}\n"
        "Dataset download karke data/raw/ folder mein telco_churn.csv naam se rakho.\n"
        "Aur script project ke root folder se run karo (churn-prediction-api)."
    )

# CSV file ko padh ke DataFrame (Excel jaisi table) mein load kiya
df = pd.read_csv(DATA_PATH)

# ---- 1. Data ka size ----
# shape (rows, columns) return karta hai. Pata chalta hai kitna data hai
print("Shape (rows, columns):", df.shape)

# ---- 2. Pehli 5 rows ----
# Data kaisa dikhta hai, ye dekhne ke liye
print("\nPehli 5 rows:")
print(df.head())

# ---- 3. Column ke data types ----
# Dekhna hai kaunsa column number hai aur kaunsa text (object)
print("\nColumn info:")
df.info()  # info() khud print karta hai, isliye print() ke andar nahi likhte

# ---- 4. Missing values ----
# isnull() har cell ko True/False banata hai, sum() True gino karta hai
print("\nMissing values per column:")
print(df.isnull().sum())

# ---- 5. Target column (jo predict karna hai) ----
# value_counts() batata hai kitne "Yes" aur kitne "No"
print("\nChurn counts:")
print(df["Churn"].value_counts())

# normalize=True se counts percentage (0 se 1) mein aa jate hain, *100 se % ban gaya
print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# ---- 6. Hidden problem check ----
# TotalCharges number hona chahiye, lekin kabhi-kabhi khali space " " hota hai.
# .str.strip() extra spaces hata deta hai, phir check karte hain string khali hai ya nahi
blank_total_charges = (df["TotalCharges"].str.strip() == "").sum()
print("\nTotalCharges mein blank values:", blank_total_charges)
