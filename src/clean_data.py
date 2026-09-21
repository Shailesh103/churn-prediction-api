# pandas se data table handle karenge
import pandas as pd
# Path se folder/file ka path safe tarike se banate hain
from pathlib import Path

# Constants: input aur output ki jagah
RAW_PATH = "data/raw/telco_churn.csv"
CLEAN_PATH = "data/processed/telco_clean.csv"

def clean_data(df):
    """Raw DataFrame lo, clean DataFrame wapas do."""

    # Original DataFrame ko safe rakhne ke liye uski copy pe kaam karte hain
    df = df.copy()

    # 1. Duplicate check customerID ke saath hi karo (ID hatane se PEHLE),
    # warna alag-alag customers galti se "duplicate" ban jate hain
    df = df.drop_duplicates()

    # 2. customerID sirf ek unique ID hai, model ko isse kuch seekhne ko nahi milta
    df = df.drop(columns=["customerID"])

    # 3. TotalCharges ko number mein badlo, jo number nahi ban sakta (" ") wo NaN ban jata hai
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 4. NaN wale naye customers hain (tenure = 0), unhone kuch pay nahi kiya, isliye 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 5. Target: "Yes" -> 1 (chala gaya), "No" -> 0 (ruka hua hai)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def main():
    # Raw data padho
    raw_df = pd.read_csv(RAW_PATH)
    print("Raw shape:", raw_df.shape)

    # Cleaning function chalao
    clean_df = clean_data(raw_df)
    print("Clean shape:", clean_df.shape)

    # Sanity checks: agar kuch galat hua toh program yahin ruk jayega
    assert clean_df.isnull().sum().sum() == 0, "Abhi bhi missing values hain!"
    assert clean_df["TotalCharges"].dtype == "float64", "TotalCharges number nahi bana!"
    assert set(clean_df["Churn"].unique()) == {0, 1}, "Churn sirf 0/1 hona chahiye!"

    # Output folder na ho toh bana do, phir CSV save karo (index=False: extra 0,1,2.. column nahi)
    Path(CLEAN_PATH).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_PATH, index=False)
    print("Saved:", CLEAN_PATH)

    # Ek nazar dekhne ke liye
    print("\nTotalCharges dtype:", clean_df["TotalCharges"].dtype)
    print("Churn values:\n", clean_df["Churn"].value_counts())


# Ye line ka matlab: file ko seedha run kiya ho tabhi main() chale.
# Baad mein API isse import karegi toh ye apne aap nahi chalega.
if __name__ == "__main__":
    main()