# pandas: data table handle karne ke liye
import pandas as pd
# numpy: yahan sirf NaN (khali value) ke liye chahiye
import numpy as np
# Path: folder/file ka path safe tarike se banane ke liye
from pathlib import Path

# Input (Step 2 ka output) aur output ki jagah
CLEAN_PATH = "data/processed/telco_clean.csv"
FEATURES_PATH = "data/processed/telco_features.csv"

# Wo 6 extra services jinko ginkar hum "customer kitna engaged hai" nikalenge
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_features(df):
    """Clean DataFrame lo, naye columns (features) jodkar wapas do."""

    # Original ko safe rakhne ke liye copy
    df = df.copy()

    # 1. tenure_group: mahino ko 4 groups mein todna.
    # bins=[-1, 12, 24, 48, 72] ka matlab: (-1 se 12], (12 se 24], (24 se 48], (48 se 72]
    # -1 isliye ki tenure=0 wale customers bhi pehle group mein aa jayein
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
    ).astype(str)  # astype(str): group ko normal text bana diya

    # 2. num_addon_services: 6 add-on columns mein kitne "Yes" hain
    # (df[...] == "Yes") har cell ko True/False banata hai,
    # sum(axis=1) har row ke True gin deta hai (axis=1 = row ke hisaab se)
    df["num_addon_services"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)

    # 3. is_month_to_month: contract month-to-month hai toh 1, warna 0
    # astype(int) True/False ko 1/0 mein badalta hai
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # 4. is_auto_pay: PaymentMethod mein "automatic" likha hai toh 1, warna 0
    df["is_auto_pay"] = df["PaymentMethod"].str.contains("automatic").astype(int)

    # 5. avg_monthly_charge: TotalCharges / tenure
    # tenure=0 ko NaN kar diya taaki 0 se divide na ho, phir NaN ki jagah MonthlyCharges bhar diya
    avg = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["avg_monthly_charge"] = avg.fillna(df["MonthlyCharges"])

    return df


def main():
    # Step 2 ka clean data padho
    df = pd.read_csv(CLEAN_PATH)
    print("Pehle:", df.shape)

    # Features jodo
    feat_df = add_features(df)
    print("Baad mein:", feat_df.shape)

    # Safety check: koi missing value nahi honi chahiye
    assert feat_df.isnull().sum().sum() == 0, "Missing values aa gayi!"

    # Save karo
    Path(FEATURES_PATH).parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(FEATURES_PATH, index=False)

    # Har naye feature ka churn % dikhao, taaki pata chale wo kaam ka hai ya nahi
    # groupby: group banao, ["Churn"].mean(): us group mein churn ka average (0/1 ka average = %)
    for col in ["tenure_group", "num_addon_services", "is_month_to_month", "is_auto_pay"]:
        print(f"\n--- {col} ---")
        print((feat_df.groupby(col)["Churn"].mean() * 100).round(1))


# Sirf seedha run karne pe main() chale, import karne pe nahi
if __name__ == "__main__":
    main()