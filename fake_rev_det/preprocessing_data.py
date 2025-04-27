import sys
import os
import pandas as pd
from utils import clean_text

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def preprocess_and_combine():
    # Load datasets
    shorter_dataset = pd.read_excel("D:/Soft Dev LAB/fake_rev_det/shorter_dataset.xlsx", engine="openpyxl")
    shorter_dataset = shorter_dataset[["Review", "Real"]].rename(columns={
        "Review": "review", 
        "Real": "label"
    })
    
    bigger_dataset = pd.read_excel("D:/Soft Dev LAB/fake_rev_det/bigger_dataset.xlsx", engine="openpyxl")
    bigger_dataset = bigger_dataset[["reviewContent", "flagged"]].rename(columns={
        "reviewContent": "review", 
        "flagged": "label"
    })

    # Split into fake/real
    bigger_fake = bigger_dataset[bigger_dataset["label"] == 0]
    bigger_real = bigger_dataset[bigger_dataset["label"] == 1]

    # Combine all fake and real reviews
    all_fake = pd.concat([
        shorter_dataset[shorter_dataset["label"] == 1], 
        bigger_fake
    ], ignore_index=True)
    
    all_real = pd.concat([
        shorter_dataset[shorter_dataset["label"] == 0], 
        bigger_real
    ], ignore_index=True)

    # Balance with replacement
    real_balanced = all_real.sample(n=len(all_fake), replace=True, random_state=42)
    combined_data = pd.concat([all_fake, real_balanced], ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean text with enhanced validation
    combined_data["cleaned_review"] = combined_data["review"].apply(clean_text)
    
    # Remove invalid entries
    combined_data = combined_data.dropna(subset=["cleaned_review"])
    combined_data = combined_data[combined_data["cleaned_review"].str.strip().astype(bool)]
    combined_data["cleaned_review"] = combined_data["cleaned_review"].astype(str)

    return combined_data

if __name__ == "__main__":
    data = preprocess_and_combine()
    data.to_csv("combined_data.csv", index=False)
    print("Preprocessing complete! Saved to combined_data.csv.")
    print(f"Final dataset size: {len(data)} rows")
    print("Sample cleaned reviews:")
    print(data[["review", "cleaned_review"]].head())