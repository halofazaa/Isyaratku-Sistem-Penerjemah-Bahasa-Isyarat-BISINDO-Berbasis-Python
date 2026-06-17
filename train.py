import pandas as pd
import glob
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if not os.path.exists('models'): os.makedirs('models')

print("1. Membaca Data...")
all_files = glob.glob("data/*.csv")
if not all_files:
    print("ERROR: Data kosong! Jalankan collect.py dulu.")
    exit()

df_list = []
for f in all_files:
    # Membaca CSV tanpa header
    df = pd.read_csv(f, header=None)
    df_list.append(df)

full_df = pd.concat(df_list, axis=0, ignore_index=True)

# Memisahkan Label (Kolom 0) dan Fitur (Kolom 1 sampai 84)
X = full_df.iloc[:, 1:].values
y = full_df.iloc[:, 0].values

# Cek konsistensi data
print(f"Jumlah Sampel: {len(X)}")
print(f"Jumlah Fitur: {X.shape[1]} (Harus 84)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Melatih Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42) # n_estimators standar agar file tidak berat
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"3. Selesai! Akurasi: {acc*100:.2f}%")

with open('models/bisindo_model.pkl', 'wb') as f:
    pickle.dump(model, f)