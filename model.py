import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import joblib
import json

df = pd.read_csv("data.csv")
df.head()

FEATURE_COLS = [
    'temperature', 'pressure', 'vibration',
    'energy_consumption', 'humidity', 'anomaly_flag'
]

X = df.drop(columns=['timestamp', 'machine_id', 'predicted_remaining_life',
                      'failure_type', 'downtime_risk', 'maintenance_required', 'machine_status'])



feature_names = list(X.columns)

y_maintenance = df['maintenance_required'].to_numpy(dtype=np.float32)

scaler = StandardScaler()
X_scaled = np.array(scaler.fit_transform(X), dtype=np.float32)

X_temp, X_test, y_temp, y_test = train_test_split( #70, 15, 15
    X_scaled, y_maintenance, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42)



class DatasetLoader(Dataset):  #batch data to 64 rows
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(DatasetLoader(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(DatasetLoader(X_val,   y_val),   batch_size=64, shuffle=False)
test_loader  = DataLoader(DatasetLoader(X_test,  y_test),  batch_size=64, shuffle=False)



class BinaryClassifierANN(nn.Module):
    def __init__(self, input_dim, hidden_size1):
        super(BinaryClassifierANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def evaluate_model(model, data_loader, loss_fn): # loss, accuracy
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch).squeeze()
            loss = loss_fn(outputs, y_batch)
            preds = (outputs > 0.5).float()
            total_loss += loss.item()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return total_loss / len(data_loader), correct / total


def train_with_eval(model, optimizer, loss_fn,
                    train_loader, val_loader, test_loader, num_epochs):
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = loss_fn(outputs, y_batch)
            preds = (outputs > 0.5).float()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        val_loss,  val_accuracy  = evaluate_model(model, val_loader,  loss_fn)
        test_loss, test_accuracy = evaluate_model(model, test_loader, loss_fn)

        train_losses.append(avg_train_loss);  val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy);  test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_accuracy:.4f}")

    return (train_losses, val_losses, test_losses,
            train_accuracies, val_accuracies, test_accuracies)


input_dim = X_train.shape[1] #train
model_maintenance = BinaryClassifierANN(input_dim, hidden_size1=32)
pos_weight = torch.tensor([4.0])
bce = nn.BCELoss(weight=None)  
bce = nn.BCELoss()

class WeightedBCELoss(nn.Module): #بزود وزن لل missed maintenance
    def __init__(self, pos_weight):
        super().__init__()
        self.pw = pos_weight
    def forward(self, pred, target):
        weights = torch.where(target == 1, self.pw, torch.ones_like(target))
        return nn.functional.binary_cross_entropy(pred, target, weight=weights)

bce = WeightedBCELoss(pos_weight)
optimizer = torch.optim.Adam(model_maintenance.parameters(), lr=0.0001)

(train_losses, val_losses, test_losses,
 train_accuracies, val_accuracies, test_accuracies) = train_with_eval(
    model_maintenance, optimizer, bce,
    train_loader, val_loader, test_loader,
    num_epochs=50
)



# names, shapes, threshold in json
data = {
    "feature_names": feature_names,
    "input_dim":     input_dim,
    "hidden_size1":  32,
}
data["threshold"] = 0.35  
with open("model_meta.json", "w") as f:
    json.dump(data, f, indent=2)
print("model_meta saved")

# trained weights in json
sd = model_maintenance.state_dict()
weights_json = {
    "fc1_weight": sd["model.0.weight"].tolist(),
    "fc1_bias":   sd["model.0.bias"].tolist(),
    "fc2_weight": sd["model.3.weight"].tolist(),
    "fc2_bias":   sd["model.3.bias"].tolist(),
}
with open("model_weights.json", "w") as f:
    json.dump(weights_json, f)
print("model_weights saved")

# scaler mean, std in json
scaler_json = {
    "mean_":  scaler.mean_.tolist(),
    "scale_": scaler.scale_.tolist(),
}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_json, f)
print("scaler_params saved")

print("\nAll required files saved")
