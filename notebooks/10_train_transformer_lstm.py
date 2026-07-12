import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os
import pickle

# ── Settings ──────────────────────────────────────────────
PROCESSED_FOLDER = "../data/processed/"
MODELS_FOLDER    = "../models/"
os.makedirs(MODELS_FOLDER, exist_ok=True)

SEQUENCE_LEN  = 20
BATCH_SIZE    = 16
EPOCHS        = 50
LEARNING_RATE = 0.0005
EARLY_STOP    = 10

FEATURE_COLS = [
    "SMA_5", "SMA_10", "SMA_20",
    "EMA_12", "EMA_26",
    "MACD", "MACD_signal", "MACD_hist",
    "RSI",
    "Return_5", "Return_10",
    "BB_upper", "BB_lower", "BB_mid", "BB_width",
    "Volatility_20", "ATR",
    "Volume_change", "Volume_MA_10",
    "news_sentiment", "news_count",
    "wsb_sentiment", "wsb_avg_score",
    "combined_sentiment"
]

TARGET_COL = "Target"
# ──────────────────────────────────────────────────────────

# ── Load data ─────────────────────────────────────────────
print("Loading final dataset...")
df = pd.read_csv(os.path.join(PROCESSED_FOLDER, "final_dataset.csv"))
df["Date"] = pd.to_datetime(df["Date"])

train_df = df[(df["Date"] >= "2021-01-01") & (df["Date"] < "2023-01-01")]
val_df   = df[(df["Date"] >= "2023-01-01") & (df["Date"] < "2024-01-01")]
test_df  = df[df["Date"] >= "2024-01-01"]

print(f"Train : {len(train_df)} rows")
print(f"Val   : {len(val_df)} rows")
print(f"Test  : {len(test_df)} rows")

# ── Scale features ────────────────────────────────────────
print("\nScaling features...")
scaler = StandardScaler()
train_df = train_df.copy()
val_df   = val_df.copy()
test_df  = test_df.copy()

train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
val_df[FEATURE_COLS]   = scaler.transform(val_df[FEATURE_COLS])
test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

# ── Build sequences ───────────────────────────────────────
def build_sequences(data, seq_len=SEQUENCE_LEN):
    sequences = []
    targets   = []
    for ticker, group in data.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        X = group[FEATURE_COLS].values
        y = group[TARGET_COL].values
        for i in range(seq_len, len(group)):
            sequences.append(X[i-seq_len:i])
            targets.append(y[i])
    return np.array(sequences), np.array(targets)

print("\nBuilding sequences (20 day windows)...")
X_train, y_train = build_sequences(train_df)
X_val,   y_val   = build_sequences(val_df)
X_test,  y_test  = build_sequences(test_df)

print(f"Train sequences : {X_train.shape}")
print(f"Val sequences   : {X_val.shape}")
print(f"Test sequences  : {X_test.shape}")

# ── PyTorch Dataset ───────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(StockDataset(X_val, y_val),
                          batch_size=BATCH_SIZE)
test_loader  = DataLoader(StockDataset(X_test, y_test),
                          batch_size=BATCH_SIZE)

# ── Model Architecture ────────────────────────────────────
class TransformerLSTM(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=4,
                 num_layers=1, lstm_hidden=32, dropout=0.5):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.input_norm       = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(lstm_hidden, 16)
        self.fc2     = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()

# ── Initialize model ──────────────────────────────────────
input_size = len(FEATURE_COLS)
model      = TransformerLSTM(input_size=input_size)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)

# ── Training loop ─────────────────────────────────────────
print("\nTraining Transformer + LSTM...")
print(f"{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<12}{'Val AUC'}")
print("-" * 46)

best_val_auc     = 0
best_model_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_losses = []

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # Validation phase
    model.eval()
    val_losses = []
    val_preds  = []
    val_true   = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            val_losses.append(loss.item())
            val_preds.extend(preds.numpy())
            val_true.extend(y_batch.numpy())

    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)
    val_auc    = roc_auc_score(val_true, val_preds)

    # Save best model
    if val_auc > best_val_auc:
        best_val_auc     = val_auc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    scheduler.step(val_loss)
    print(f"{epoch+1:<8}{train_loss:<14.4f}{val_loss:<12.4f}{val_auc:.4f}")

    # Early stopping
    if patience_counter >= EARLY_STOP:
        print(f"\nEarly stopping at epoch {epoch+1} — no improvement for {EARLY_STOP} epochs")
        break

# ── Evaluate on test set ──────────────────────────────────
print(f"\nBest validation AUC: {best_val_auc:.4f}")
print("\nEvaluating on test set...")
model.load_state_dict(best_model_state)
model.eval()

test_preds = []
test_true  = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        test_preds.extend(preds.numpy())
        test_true.extend(y_batch.numpy())

test_preds_binary = [1 if p > 0.5 else 0 for p in test_preds]
test_accuracy     = accuracy_score(test_true, test_preds_binary)
test_auc          = roc_auc_score(test_true, test_preds)

print(f"\n── Test Results (2024) ──────────────────────────────")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"AUC ROC  : {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(test_true, test_preds_binary))

# ── Save model and predictions ────────────────────────────
print("\nSaving model...")
torch.save(best_model_state,
           os.path.join(MODELS_FOLDER, "transformer_lstm.pt"))

with open(os.path.join(MODELS_FOLDER, "scaler_lstm.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save predictions for backtesting
test_df_copy = test_df.copy()
test_df_copy = test_df_copy.groupby("Ticker", group_keys=False).apply(
    lambda g: g.iloc[SEQUENCE_LEN:]
).reset_index(drop=True)

test_df_copy["lstm_prob"] = test_preds
test_df_copy["lstm_pred"] = test_preds_binary
test_df_copy[["Date", "Ticker", "Close", TARGET_COL,
              "lstm_prob", "lstm_pred"]].to_csv(
    os.path.join(PROCESSED_FOLDER, "lstm_predictions.csv"), index=False)

print("\n✅ Transformer + LSTM training complete!")
print(f"   Model saved       → models/transformer_lstm.pt")
print(f"   Predictions saved → data/processed/lstm_predictions.csv")