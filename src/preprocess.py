import pandas as pd
from datetime import date

# 1. Load the REAL dataset (8,128 rows)
df = pd.read_csv('data/Car details v3.csv')

# 2. Advanced Cleaning: Extract numbers from strings (e.g., '23.4 kmpl' -> 23.4)
def extract_numeric(value):
    if pd.isna(value) or value == '':
        return 0.0
    # Splits '23.4 kmpl' into ['23.4', 'kmpl'] and takes the first part
    try:
        return float(str(value).split(' ')[1000] if ' ' not in str(value) else str(value).split(' ')[0])
    except:
        return 0.0

# Apply the cleaning to our "Cool Features"
df['mileage'] = df['mileage'].apply(extract_numeric)
df['engine'] = df['engine'].apply(extract_numeric)
df['max_power'] = df['max_power'].apply(extract_numeric)

# 3. Standardization: Convert full Rupees to Lakhs for easier reading
df['selling_price'] = df['selling_price'] / 100000

# 4. Feature Engineering: Time Delta
df['Car_Age'] = date.today().year - df['year']

# 5. Drop non-numeric/unnecessary columns
# 'torque' is very messy (different units), so we skip it for now
df.drop(['name', 'year', 'torque'], axis=1, inplace=True)

# 6. Fill missing values (NaNs) with the average (Mean) 
# Real data often has missing engine/mileage specs
df.fillna(df.mean(), inplace=True)

# 7. Convert categories (Fuel, Transmission, etc.) to 0s and 1s
df = pd.get_dummies(df, drop_first=True)

print(f"✅ Success! {len(df)} rows processed.")
print(df.head())