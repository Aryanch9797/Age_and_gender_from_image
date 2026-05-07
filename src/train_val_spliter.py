import pandas as pd
from sklearn.model_selection import train_test_split

def train_val_spliter(dataset, test_size):
    dataset["stratify_col"] = (dataset['gender'].astype(str) + "_" + dataset['age'].astype(str))

    # To prevent only 1 member while using train_test_split
    class_counts = dataset['stratify_col'].value_counts()
    rare_classes = class_counts[class_counts == 1].index
    freq_classes = class_counts[class_counts >= 2].index
    rare_df = dataset[dataset['stratify_col'].isin(rare_classes)]
    freq_df = dataset[dataset['stratify_col'].isin(freq_classes)]
    
    train_freq, val_df = train_test_split(freq_df, test_size=test_size, random_state=42, stratify=freq_df['stratify_col'])
    train_df = pd.concat([train_freq, rare_df], ignore_index=True)

    print(f"train shape:{train_df.shape}")
    print(f"validation shape:{val_df.shape}")
    
    return train_df, val_df
