import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import date

# 1. Load the REAL dataset
df = pd.read_csv('data/Car details v3.csv')

# 2. Advanced Cleaning (Handling the 'kmpl', 'CC', and 'bhp' strings)
def extract_numeric(value):
    if pd.isna(value) or str(value).strip() == "": 
        return 0.0
    try:
        # Splits '23.4 kmpl' and takes only '23.4'
        return float(str(value).split(' ')[0])
    except:
        return 0.0

df['mileage'] = df['mileage'].apply(extract_numeric)
df['engine'] = df['engine'].apply(extract_numeric)
df['max_power'] = df['max_power'].apply(extract_numeric)

# 3. Standardization & Engineering
df['selling_price'] = df['selling_price'] / 100000 # Convert to Lakhs
df['Car_Age'] = date.today().year - df['year']

# 4. Drop unnecessary columns
# We drop 'name' (too many unique values) and 'torque' (messy formatting)
df.drop(['name', 'year', 'torque'], axis=1, inplace=True)

# 5. Handle Categorical Data (Fuel, Transmission, etc.)
df = pd.get_dummies(df, drop_first=True)

# 6. Split the Data
X = df.drop('selling_price', axis=1)
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the "Big Brain"
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Verification (Accuracy Score)
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

# 9. SAVE THE MODEL AND THE FEATURE MAP
# This is the secret sauce to fixing the "Shape Mismatch"
model_data = {
    "model": model,
    "features": list(X.columns) # Saves the exact column order
}

with open('models/car_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"✅ Success! Training Complete.")
print(f"📊 Accuracy (R² Score): {accuracy:.4f}")
print(f"📐 Features Learned: {len(X.columns)}")