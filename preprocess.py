import pandas as pd

def load_and_preprocess(path):
    # dataset load చేయటం
    df = pd.read_csv(path)
    
    print("✅ Original Dataset Shape:", df.shape)
    print(df.head())

    # Missing values ఉంటే handle చేయటం
    df = df.dropna()

    # అవసరమైతే categorical columns → numeric (label encoding, one-hot encoding)
    # ఉదాహరణ: df['Crop'] = df['Crop'].astype('category').cat.codes

    print("✅ Cleaned Dataset Shape:", df.shape)
    return df
