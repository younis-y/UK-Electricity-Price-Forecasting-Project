#!/usr/bin/env python3
"""
04_train_additional_dl_models.py
=================================
Trains DNN, LSTM, and BiLSTM models using the same data pipeline and
evaluation protocol as 03_price_prediction_main.py.

Run from the repo root:
    python scripts/04_train_additional_dl_models.py

Results are appended to data/training_log.jsonl and printed to stdout.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import holidays

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ''))
import training_logger

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — must match 03_price_prediction_main.py exactly
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
WINDOW       = 168
BATCH_SIZE   = 256
MAX_EPOCHS   = 500
PATIENCE     = 40
WARMUP_PCT   = 0.1

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f"PyTorch {torch.__version__} | Device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (mirrors 03_price_prediction_main.py §4.1 – §4.4)
# ──────────────────────────────────────────────────────────────────────────────
def find_file(name, alt_name=None):
    for prefix in ['data/raw/', 'data/processed/', 'data/predictions/',
                   '../data/raw/', '../data/processed/', '../data/predictions/', '']:
        for n in [name] + ([alt_name] if alt_name else []):
            if n and os.path.exists(prefix + n):
                return prefix + n
    raise FileNotFoundError(f'{name} not found')

print("\nLoading raw data ...")
df_prices   = pd.read_csv(find_file('elec_price_hourly_entsoe.csv'))
df_wind_p   = pd.read_csv(find_file('wind_gen_predicted_hourly_xgboost.csv'))
df_solar_p  = pd.read_csv(find_file('solar_gen_predicted_hourly_xgboost.csv'))
df_demand   = pd.read_csv(find_file('elec_demand_outturn_hh_bmrs.csv'))
df_gas      = pd.read_csv(find_file('gas_sap_daily_icis.csv'))
df_co2      = pd.read_csv(find_file('carbon_ukets_futures_daily_investing.csv'))

# Optional DA datasets
try:
    df_demand_da  = pd.read_csv(find_file('elec_demand_forecast_da_hh_bmrs.csv'))
    HAS_DA_DEMAND = True
except FileNotFoundError:
    HAS_DA_DEMAND = False

try:
    df_ws_da       = pd.read_csv(find_file('renew_gen_forecast_da_hourly_bmrs.csv'))
    HAS_DA_WINDSOLAR = True
except FileNotFoundError:
    HAS_DA_WINDSOLAR = False

DEMAND_IS_PERSISTENCE_FORECAST = False

def strip_tz(idx):
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_convert(None)
    return idx

# ---------- Demand ----------
if HAS_DA_DEMAND:
    df_demand_da['datetime'] = pd.to_datetime(df_demand_da['datetime'])
    df_demand_da.set_index('datetime', inplace=True)
    df_demand_da.index = strip_tz(df_demand_da.index)
    df_demand_da_hourly = df_demand_da[['Demand_DA_MW']].resample('h').mean().rename(
        columns={'Demand_DA_MW': 'Demand_MW'})
    da_days = (df_demand_da_hourly.index.max() - df_demand_da_hourly.index.min()).days
    if da_days < 30:
        HAS_DA_DEMAND = False
    else:
        df_demand_hourly = df_demand_da_hourly
        print(f"  -> Using DA demand forecast ({da_days} days)")

if not HAS_DA_DEMAND:
    df_demand['Datetime'] = pd.to_datetime(df_demand['SETTLEMENT_DATE'], dayfirst=True, format='mixed') + \
                            pd.to_timedelta((df_demand['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')
    df_demand.set_index('Datetime', inplace=True)
    df_demand_outturn_hourly = df_demand[['ND']].resample('h').mean().rename(columns={'ND': 'Demand_MW'})
    df_demand_hourly = df_demand_outturn_hourly.shift(24).dropna()
    DEMAND_IS_PERSISTENCE_FORECAST = True

# ---------- Wind / Solar predictions ----------
df_wind_p['timestamp'] = pd.to_datetime(df_wind_p['timestamp'])
df_wind_p.set_index(df_wind_p['timestamp'].dt.tz_convert(None), inplace=True)
df_wind_hourly = df_wind_p[['predicted_mw']].resample('h').mean().rename(
    columns={'predicted_mw': 'Wind_Predicted_MW'})

df_solar_p['timestamp'] = pd.to_datetime(df_solar_p['timestamp'])
df_solar_p.set_index(df_solar_p['timestamp'].dt.tz_convert(None), inplace=True)
df_solar_hourly = df_solar_p[['predicted_mw']].resample('h').mean().rename(
    columns={'predicted_mw': 'Solar_Predicted_MW'})

# ---------- BMRS DA wind/solar ----------
if HAS_DA_WINDSOLAR:
    df_ws_da['datetime'] = pd.to_datetime(df_ws_da['datetime'])
    df_ws_da.set_index('datetime', inplace=True)
    df_ws_da.index = strip_tz(df_ws_da.index)
    ws_da_cols = [c for c in df_ws_da.columns if 'Forecast' in c]
    df_ws_da_hourly = df_ws_da[ws_da_cols].resample('h').mean()
    ws_da_days = (df_ws_da_hourly.index.max() - df_ws_da_hourly.index.min()).days
    if ws_da_days < 30:
        HAS_DA_WINDSOLAR = False
        df_ws_da_hourly = pd.DataFrame()
else:
    df_ws_da_hourly = pd.DataFrame()

# ---------- Prices ----------
df_prices['Datetime'] = pd.to_datetime(df_prices['Datetime (UTC)'], dayfirst=True, format='mixed')
df_prices.set_index('Datetime', inplace=True)
df_prices_hourly = df_prices[['Price (EUR/MWhe)']].sort_index().rename(
    columns={'Price (EUR/MWhe)': 'Price_EUR'})

# ---------- Gas — lag 1 day ----------
df_gas['Date'] = pd.to_datetime(df_gas['Date'], dayfirst=True, format='mixed')
df_gas = df_gas.sort_values('Date').set_index('Date')
gas_col = 'SAP actual day' if 'SAP actual day' in df_gas.columns else df_gas.columns[0]
df_gas_daily = df_gas[[gas_col]].rename(columns={gas_col: 'Gas_Price'}).shift(1)
df_gas_hourly = df_gas_daily.reindex(
    pd.date_range(df_gas.index.min(), df_gas.index.max(), freq='D')
).interpolate().resample('h').ffill()

# ---------- CO2 — lag 1 day ----------
df_co2['Date'] = pd.to_datetime(df_co2['Date'], dayfirst=True, format='mixed')
df_co2 = df_co2.sort_values('Date').set_index('Date')
df_co2_daily = df_co2[['Price']].rename(columns={'Price': 'CO2_Price'}).shift(1)
df_co2_hourly = df_co2_daily.reindex(
    pd.date_range(df_co2.index.min(), df_co2.index.max(), freq='D')
).interpolate().resample('h').ffill()

# ---------- Merge ----------
data = df_prices_hourly.join(df_demand_hourly, how='inner')
data = data.join(df_wind_hourly, how='inner')
data = data.join(df_solar_hourly, how='inner')
data = data.join(df_gas_hourly, how='inner')
data = data.join(df_co2_hourly, how='inner')

if HAS_DA_WINDSOLAR and not df_ws_da_hourly.empty:
    data = data.join(df_ws_da_hourly, how='left')

# ---------- Feature Engineering (mirrors main script §4.3) ----------
data['Residual_Load']       = data['Demand_MW'] - (data['Wind_Predicted_MW'] + data['Solar_Predicted_MW'])
data['Theoretical_Cost']    = data['Gas_Price'] + (0.5 * data['CO2_Price'])
data['Cost_Load_Interaction']= data['Theoretical_Cost'] * data['Residual_Load']
data['ResLoad_Roll_Mean_24'] = data['Residual_Load'].rolling(window=24, closed='left').mean()
data['ResLoad_Roll_Std_24']  = data['Residual_Load'].rolling(window=24, closed='left').std()

uk_holidays = holidays.UnitedKingdom(years=list(range(2021, 2026)))
data['is_holiday'] = data.index.map(lambda x: 1 if x in uk_holidays else 0)
data['hour_sin']   = np.sin(2 * np.pi * data.index.hour / 24)
data['hour_cos']   = np.cos(2 * np.pi * data.index.hour / 24)
data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
data['Price_Lag_24']       = data['Price_EUR'].shift(24)
data['Price_Lag_168']      = data['Price_EUR'].shift(168)
data['Price_Roll_Mean_24'] = data['Price_EUR'].rolling(window=24, closed='left').mean()

if 'Wind_Forecast_DA_MW' in data.columns:
    data['Wind_Forecast_Error_XGB']  = data['Wind_Predicted_MW']  - data['Wind_Forecast_DA_MW']
if 'Solar_Forecast_DA_MW' in data.columns:
    data['Solar_Forecast_Error_XGB'] = data['Solar_Predicted_MW'] - data['Solar_Forecast_DA_MW']
if 'Wind_Forecast_DA_MW' in data.columns and 'Solar_Forecast_DA_MW' in data.columns:
    data['Residual_Load_DA'] = data['Demand_MW'] - (data['Wind_Forecast_DA_MW'] + data['Solar_Forecast_DA_MW'])

data.dropna(inplace=True)

# ---------- Feature list (mirrors main script) ----------
feature_cols = [
    'Residual_Load', 'ResLoad_Roll_Mean_24', 'ResLoad_Roll_Std_24',
    'Gas_Price', 'CO2_Price', 'Theoretical_Cost', 'Cost_Load_Interaction',
    'Wind_Predicted_MW', 'Solar_Predicted_MW', 'Demand_MW',
    'Price_Lag_24', 'Price_Lag_168', 'Price_Roll_Mean_24',
    'hour_sin', 'hour_cos', 'is_holiday', 'is_weekend'
]
da_features = ['Wind_Forecast_DA_MW', 'Solar_Forecast_DA_MW',
               'Wind_Forecast_Error_XGB', 'Solar_Forecast_Error_XGB', 'Residual_Load_DA']
for f in da_features:
    if f in data.columns:
        feature_cols.append(f)

print(f"\nDataset: {len(data):,} rows | Features: {len(feature_cols)}")

# ---------- Train / test split (80/20) ----------
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data  = data.iloc[train_size:]

X_train_price = train_data[feature_cols]
y_train_price = train_data['Price_EUR']
X_test_price  = test_data[feature_cols]
y_test_price  = test_data['Price_EUR']

print(f"Train: {len(train_data):,} rows  "
      f"({train_data.index.min().date()} → {train_data.index.max().date()})")
print(f"Test:  {len(test_data):,} rows   "
      f"({test_data.index.min().date()} → {test_data.index.max().date()})")

# ---------- Scaling ----------
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train_price)
X_test_sc  = scaler_X.transform(X_test_price)
y_train_sc = scaler_y.fit_transform(y_train_price.values.reshape(-1, 1)).ravel()

# ---------- Sequence windows ----------
def create_sequences(X, y, window):
    n = len(X) - window
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    return X[idx], y[window:]

X_seq_all, y_seq_all = create_sequences(X_train_sc, y_train_sc, WINDOW)

X_combined = np.vstack([X_train_sc[-WINDOW:], X_test_sc])
y_combined  = np.concatenate([
    y_train_sc[-WINDOW:],
    scaler_y.transform(y_test_price.values.reshape(-1, 1)).ravel()
])
X_seq_test, _ = create_sequences(X_combined, y_combined, WINDOW)

val_size  = int(len(X_seq_all) * 0.1)
X_seq_tr  = X_seq_all[:-val_size];  y_seq_tr  = y_seq_all[:-val_size]
X_seq_val = X_seq_all[-val_size:];  y_seq_val = y_seq_all[-val_size:]

X_tr_t  = torch.FloatTensor(X_seq_tr);  y_tr_t  = torch.FloatTensor(y_seq_tr)
X_val_t = torch.FloatTensor(X_seq_val); y_val_t = torch.FloatTensor(y_seq_val)
X_te_t  = torch.FloatTensor(X_seq_test)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

n_features = X_train_sc.shape[1]
print(f"\nSequence shapes  train={X_seq_tr.shape}  val={X_seq_val.shape}  test={X_seq_test.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# ──────────────────────────────────────────────────────────────────────────────

class DNNModel(nn.Module):
    """5-layer feedforward DNN.

    Takes only the LAST time step from the (batch, 168, F) input tensor,
    making it a true non-temporal baseline: any improvement by LSTM/TCN
    over this model is attributable to learned temporal representations.
    """
    def __init__(self, input_size, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 256),        nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):          # x: (B, T, F)
        x = x[:, -1, :]           # take last timestep → (B, F)
        return self.net(x).squeeze(-1)


class TemporalAttention(nn.Module):
    """Additive attention over hidden states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H):           # H: (B, T, hidden_dim)
        scores  = self.v(torch.tanh(self.W(H)))   # (B, T, 1)
        weights = torch.softmax(scores, dim=1)     # (B, T, 1)
        context = (weights * H).sum(dim=1)         # (B, hidden_dim)
        return context


class LSTMModel(nn.Module):
    """3-layer LSTM with temporal attention."""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.attn = TemporalAttention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):          # x: (B, T, F)
        H, _ = self.lstm(x)        # H: (B, T, hidden)
        ctx  = self.attn(H)        # (B, hidden)
        return self.head(ctx).squeeze(-1)


class BiLSTMModel(nn.Module):
    """3-layer bidirectional LSTM with temporal attention."""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout,
                              bidirectional=True)
        self.attn = TemporalAttention(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):          # x: (B, T, F)
        H, _ = self.bilstm(x)     # H: (B, T, 2*hidden)
        ctx  = self.attn(H)       # (B, 2*hidden)
        return self.head(ctx).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
_DL_CONFIGS = {
    'DNN':    {'max_lr': 1e-3, 'weight_decay': 1e-3, 'grad_clip': 1.0},
    'LSTM':   {'max_lr': 1e-3, 'weight_decay': 1e-2, 'grad_clip': 1.0},
    'BiLSTM': {'max_lr': 1e-3, 'weight_decay': 1e-2, 'grad_clip': 1.0},
}


def train_dl_model(model, name):
    """Train one DL model; mirrors the training loop in 03_price_prediction_main.py."""
    cfg = _DL_CONFIGS[name]
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['max_lr'] / 25,
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['max_lr'],
        epochs=MAX_EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=WARMUP_PCT, anneal_strategy='cos')
    criterion = nn.MSELoss()

    best_val  = float('inf')
    best_state = None
    wait = 0
    Xv, yv = X_val_t.to(DEVICE), y_val_t.to(DEVICE)
    train_losses, val_losses = [], []

    print(f"  {name}: max_lr={cfg['max_lr']}, wd={cfg['weight_decay']}, "
          f"grad_clip={cfg['grad_clip']}, scheduler=OneCycleLR")

    for epoch in range(MAX_EPOCHS):
        model.train()
        run_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
            scheduler.step()
            run_loss += loss.item()

        avg_train = run_loss / len(train_loader)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()

        train_losses.append(avg_train)
        val_losses.append(val_loss)

        training_logger.log_epoch(
            name, epoch + 1, MAX_EPOCHS,
            train_loss=avg_train, val_loss=val_loss,
            lr=optimizer.param_groups[0]['lr'],
            patience_counter=wait, best_val_loss=best_val)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  {name}: Early stop @ epoch {epoch+1}  "
                      f"(best val MSE: {best_val:.6f})")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  {name}: Epoch {epoch+1}/{MAX_EPOCHS}  "
                  f"val_MSE={val_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state)
    model.eval()
    best_epoch = len(train_losses) - wait
    return model, train_losses, val_losses, best_epoch


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN & EVALUATE
# ──────────────────────────────────────────────────────────────────────────────
training_logger.log_stage("Additional DL Training (DNN, LSTM, BiLSTM)")

dl_models = {
    'DNN':    DNNModel(n_features, dropout=0.2),
    'LSTM':   LSTMModel(n_features, hidden_size=256, num_layers=3, dropout=0.2),
    'BiLSTM': BiLSTMModel(n_features, hidden_size=256, num_layers=3, dropout=0.2),
}

results = []

for name, model in dl_models.items():
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams  = {**_DL_CONFIGS[name],
                'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
                'batch_size': BATCH_SIZE, 'window': WINDOW,
                'n_params': n_params}
    training_logger.log_model_start(name, model_type='pytorch',
                                    hyperparams=hparams, category='Deep Learning')
    t0 = time.time()
    print(f"\n{'='*60}\nTraining {name}  ({n_params:,} params)\n{'='*60}")

    model, train_losses, val_losses, best_epoch = train_dl_model(model, name)

    # Inference on test set (batched)
    preds_sc = []
    with torch.no_grad():
        for i in range(0, len(X_te_t), BATCH_SIZE):
            preds_sc.append(model(X_te_t[i:i+BATCH_SIZE].to(DEVICE)).cpu())
    preds_sc = torch.cat(preds_sc).numpy()
    preds    = scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).ravel()

    # Test metrics
    r2   = r2_score(y_test_price, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test_price, preds)))
    mae  = float(mean_absolute_error(y_test_price, preds))
    dur  = time.time() - t0

    # Train / val RMSE at best epoch (scaled → unscaled via scaler_y std)
    y_std = float(scaler_y.scale_[0])
    train_rmse_scaled = float(np.sqrt(train_losses[best_epoch - 1])) * y_std
    val_rmse_scaled   = float(np.sqrt(val_losses[best_epoch - 1]))   * y_std

    row = {
        'Model':      name,
        'R2':         round(r2,   4),
        'RMSE':       round(rmse, 2),
        'MAE':        round(mae,  2),
        'Train_RMSE': round(train_rmse_scaled, 2),
        'Val_RMSE':   round(val_rmse_scaled,   2),
        'Best_Epoch': best_epoch,
        'Params':     n_params,
    }
    results.append(row)

    training_logger.log_model_done(name, r2=r2, rmse=rmse, mae=mae,
                                   duration_s=dur, category='Deep Learning')
    print(f"\n  {name} RESULTS:")
    print(f"    Test  R²={r2:.4f}  RMSE={rmse:.2f} EUR/MWh  MAE={mae:.2f} EUR/MWh")
    print(f"    Train RMSE={train_rmse_scaled:.2f}  Val RMSE={val_rmse_scaled:.2f}  "
          f"Best epoch={best_epoch}  Duration={dur:.0f}s")

# ──────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("  COMPLETE DL MODEL RESULTS SUMMARY")
print("="*70)
print(f"  {'Model':<10}  {'R²':>6}  {'RMSE':>8}  {'MAE':>8}  "
      f"{'TrainRMSE':>10}  {'ValRMSE':>9}  {'BestEpoch':>10}")
print("  " + "-"*70)
for row in results:
    print(f"  {row['Model']:<10}  {row['R2']:>6.4f}  {row['RMSE']:>8.2f}  "
          f"{row['MAE']:>8.2f}  {row['Train_RMSE']:>10.2f}  "
          f"{row['Val_RMSE']:>9.2f}  {row['Best_Epoch']:>10}")

print("\nFor reference — previously trained models:")
print(f"  {'TCN':<10}  {'0.8514':>6}  {'13.86':>8}  {'9.96':>8}")
print(f"  {'PatchTST':<10}  {'0.6742':>6}  {'20.52':>8}  {'14.80':>8}")
print(f"  {'XGB+SVR':<10}  {'0.8597':>6}  {'13.46':>8}  {'9.44':>8}")
print("="*70)
print("\nResults appended to data/training_log.jsonl")
