import pandas as pd

# 1. Load the data
try:
    df = pd.read_csv('data/car_data.csv')
    print("✅ Success! Data loaded.")
    
    # 2. Show the 'Columns' (The features we'll use)
    print("\nColumns found:", df.columns.tolist())
    
    # 3. Check for 'Missing Values'
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
except Exception as e:
    print(f"❌ Error loading data: {e}")