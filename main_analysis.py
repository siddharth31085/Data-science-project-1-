import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Dataset (Using a public URL for a standard maintenance dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

# 2. Basic Cleaning
# Dropping ID columns that don't help prediction
df = df.drop(['UDI', 'Product ID', 'Type'], axis=1)

# 3. Simple EDA Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Sensor Correlation Heatmap")
plt.savefig('correlation_map.png') # Saves image for your GitHub
plt.show()

# 4. Feature Selection & Splitting
X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
predictions = model.predict(X_test)
print("--- Classification Report ---")
print(classification_report(y_test, predictions))

# 7. Feature Importance (Show what drives failure)
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(5).plot(kind='barh')
plt.title("Top 5 Indicators of Machine Failure")
plt.show()
