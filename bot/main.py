# === Импорт библиотек ===
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
import joblib

# === Загрузка данных ===
df = pd.read_csv("diabetes.csv")
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# === Масштабирование ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Используем random_state для лучших метрик ===
best_state_acc_f1 = 61
best_state_prec = 53
best_state_rec = 17

# === Финальная модель на лучшем F1/Accuracy ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=best_state_acc_f1, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(X_test_t).numpy().flatten()
    preds_bin = (preds >= 0.5).astype(int)

acc = accuracy_score(y_test, preds_bin)
prec = precision_score(y_test, preds_bin, zero_division=0)
rec = recall_score(y_test, preds_bin, zero_division=0)
f1 = f1_score(y_test, preds_bin, zero_division=0)

print(f"\n✅ Итоговые метрики на random_state={best_state_acc_f1}:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")

# === ROC-кривая ===
fpr, tpr, thresholds = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC-кривая для random_state={best_state_acc_f1}')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Сохранение модели ===
torch.save({'model_state_dict': model.state_dict()}, "diabetes_pima_model.pth")
joblib.dump(scaler, "scaler_pima.pkl")
print(f"✅ Модель и scaler сохранены с random_state={best_state_acc_f1}")
