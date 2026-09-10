import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

# SKLEARN IMPORTE
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Customers.csv")
df["Profession"] = df["Profession"].fillna("Unknown")
df = df.drop(columns=["CustomerID"])

X = df.drop(columns=["Spending Score (1-100)"])
y_reg = df["Spending Score (1-100)"]
y_class = (df["Spending Score (1-100)"] >= 50).astype(int)

X_encoded = pd.get_dummies(
    X, columns=["Gender", "Profession"], drop_first=True
)

X_train, X_test, y_train_r, y_test_r, y_train_c, y_test_c = train_test_split(
    X_encoded, y_reg, y_class, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 2. LinReg vs KNN vs RF

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_r)
lr_rmse = np.sqrt(mean_squared_error(y_test_r, lr.predict(X_test_scaled)))

# KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=39)
knn_reg.fit(X_train_scaled, y_train_r)
knn_rmse = np.sqrt(mean_squared_error(y_test_r, knn_reg.predict(X_test_scaled)))

# Random Forest Regressor (prevent overfitting)
rf_reg = RandomForestRegressor(
    n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42
)
rf_reg.fit(X_train, y_train_r)
rf_rmse = np.sqrt(mean_squared_error(y_test_r, rf_reg.predict(X_test)))


# 3. CLASSIFICATION BENCHMARK vs BASELINE
# Majority Class Dummy Baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train_c)
dummy_acc = accuracy_score(y_test_c, dummy.predict(X_test)) * 100

# Logistic Regression
log_clf = LogisticRegression(C=0.01)
log_clf.fit(X_train_scaled, y_train_c)
log_acc = accuracy_score(y_test_c, log_clf.predict(X_test_scaled)) * 100

# KNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=25)
knn_clf.fit(X_train_scaled, y_train_c)
knn_acc = accuracy_score(y_test_c, knn_clf.predict(X_test_scaled)) * 100

# Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=42
)
rf_clf.fit(X_train, y_train_c)
rf_acc = accuracy_score(y_test_c, rf_clf.predict(X_test)) * 100


# 4. PLOTTING
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Regression Comparison
models_reg = [
    "Linear Regression",
    "KNN Regressor (k=39)",
    "Random Forest (Tuned)",
]
rmses = [lr_rmse, knn_rmse, rf_rmse]
colors_reg = ["#e74c3c", "#3498db", "#2ecc71"]

bars1 = axes[0].bar(models_reg, rmses, color=colors_reg, width=0.45)
axes[0].set_title(
    "Regression Benchmark (Test RMSE)", fontsize=12, fontweight="bold"
)
axes[0].set_ylabel("Test RMSE (Lower is Better)", fontsize=10)
axes[0].set_ylim(0, 35)
axes[0].grid(axis="y", linestyle="--", alpha=0.6)

for bar in bars1:
  yval = bar.get_height()
  axes[0].text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.8,
      f"{yval:.2f}",
      ha="center",
      va="bottom",
      fontweight="bold",
  )

# Plot 2: Classification Accuracy vs Baseline
models_clf = [
    "Dummy Baseline",
    "Logistic Regression",
    "KNN Classifier",
    "Random Forest",
]
accs = [dummy_acc, log_acc, knn_acc, rf_acc]
colors_clf = ["#95a5a6", "#e67e22", "#9b59b6", "#27ae60"]

bars2 = axes[1].bar(models_clf, accs, color=colors_clf, width=0.5)
axes[1].set_title(
    "Classification Benchmark vs Baseline", fontsize=12, fontweight="bold"
)
axes[1].set_ylabel("Accuracy (%)", fontsize=10)
axes[1].set_ylim(0, 100)
axes[1].axhline(
    y=50.0, color="black", linestyle=":", label="Random Guessing (50%)"
)
axes[1].grid(axis="y", linestyle="--", alpha=0.6)

for bar in bars2:
  yval = bar.get_height()
  axes[1].text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 2,
      f"{yval:.1f}%",
      ha="center",
      va="bottom",
      fontweight="bold",
  )

axes[1].legend()

plt.tight_layout()
plt.show()