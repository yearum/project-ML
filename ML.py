import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load data
df = pd.read_csv('heartDisease.csv', index_col=0)
print(df.head(50))

# menghapus data duplikat
df = df.drop_duplicates()

# EDA
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='usia', bins=10, kde=True, color='#A8D5BA', edgecolor="black", linewidth=0.6)
plt.title('Distribusi Usia', fontsize=16, fontweight='bold', color='#34495E')
plt.xlabel('Usia', fontsize=12, color='#34495E')
plt.ylabel('Frekuensi', fontsize=12, color='#A8D5BA')
plt.show()

if 'diagnosis' in df.columns:
    sns.set_palette("crest")
    df['diagnosis'].value_counts().plot(kind='pie', autopct='%1.5f%%')
    plt.title('Presentase penyakit jantung vs Non penyakit jantung')
    plt.show()
else:
    print("Kolom 'diagnosis' tidak ditemukan.")

# Data preparation
jk_encoder = LabelEncoder()
nd_encoder = LabelEncoder()
s_encoder = LabelEncoder()

df['jenis kelamin'] = jk_encoder.fit_transform(df['jenis kelamin'])
df['nyeri dada'] = nd_encoder.fit_transform(df['nyeri dada'])
df['slope'] = s_encoder.fit_transform(df['slope'])

df['diagnosis'] = df['diagnosis'].map({'Jantung': 1, 'Normal': 0})

if df['diagnosis'].isnull().any():
    print("Terdapat nilai NaN di kolom 'diagnosis' setelah mapping. Periksa data.")
    print(df['diagnosis'].value_counts(dropna=False))
    exit()

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Standarisasi fitur
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(kernel='linear')
}

results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"],
    })

evaluation_df = pd.DataFrame(results)
print(evaluation_df)

# Data baru untuk prediksi
new_data = pd.DataFrame({
    'usia': [57],
    'jenis kelamin': ["Perempuan"],
    'nyeri dada': ["Atypical angina"],
    'trestbps': [130],
    'cholestoral': [230],  
    'fasting blood sugar': [0],
    'restecg': [0],
    'denyut jantung': [175],
    'exang': [0],
    'oldpeak': [0.5],
    'slope': ["Flatsloping"],
    'ca': [1],
    'thalium': [2],
})

# Encode fitur kategorikal di new_data
new_data['jenis kelamin'] = jk_encoder.transform(new_data['jenis kelamin'])
new_data['nyeri dada'] = nd_encoder.transform(new_data['nyeri dada'])
new_data['slope'] = s_encoder.transform(new_data['slope'])

new_data = new_data[X.columns]

new_data_scaled = scaler.transform(new_data)

# Memprediksi menggunakan model yang telah dilatih
for model_name, model in models.items():
    prediction = model.predict(new_data_scaled)
    hasil_prediksi = "Normal" if prediction[0] == 0 else "Jantung"
    print(f"Prediksi penyakit oleh {model_name}: {hasil_prediksi}")
