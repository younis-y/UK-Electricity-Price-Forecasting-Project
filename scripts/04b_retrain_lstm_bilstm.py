#!/usr/bin/env python3
"""
04b_retrain_lstm_bilstm.py
==========================
Re-trains LSTM and BiLSTM with corrected hyperparameters:
  - max_lr = 3e-4  (1e-3 caused instability with OneCycleLR)
  - PyTorch default weight initialisation (more stable for LSTMs)
  - Forget-gate bias initialised to +1.0  (standard LSTM best practice)
  - Model names suffixed 'v2' in log to avoid overwriting original runs

Run from repo root: python scripts/04b_retrain_lstm_bilstm.py
"""

import os, sys, time, warnings, json, math
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
# DATA PIPELINE (identical to 04_train_additional_dl_models.py)
# ──────────────────────────────────────────────────────────────────────────────
def find_file(name):
    for prefix in ['data/raw/', 'data/processed/', 'data/predictions/',
                   '../data/raw/', '../data/processed/', '../data/predictions/', '']:
        if os.path.exists(prefix + name):
            return prefix + name
    raise FileNotFoundError(name)

def strip_tz(idx):
    return idx.tz_convert(None) if hasattr(idx, 'tz') and idx.tz else idx

print("\nLoading data...")
df_prices  = pd.read_csv(find_file('elec_price_hourly_entsoe.csv'))
df_wind_p  = pd.read_csv(find_file('wind_gen_predicted_hourly_xgboost.csv'))
df_solar_p = pd.read_csv(find_file('solar_gen_predicted_hourly_xgboost.csv'))
df_demand  = pd.read_csv(find_file('elec_demand_outturn_hh_bmrs.csv'))
df_gas     = pd.read_csv(find_file('gas_sap_daily_icis.csv'))
df_co2     = pd.read_csv(find_file('carbon_ukets_futures_daily_investing.csv'))

HAS_DA_DEMAND = False; HAS_DA_WINDSOLAR = False
try:
    df_demand_da = pd.read_csv(find_file('elec_demand_forecast_da_hh_bmrs.csv'))
    HAS_DA_DEMAND = True
except FileNotFoundError: pass
try:
    df_ws_da = pd.read_csv(find_file('renew_gen_forecast_da_hourly_bmrs.csv'))
    HAS_DA_WINDSOLAR = True
except FileNotFoundError: pass

DEMAND_IS_PERSISTENCE = False

if HAS_DA_DEMAND:
    df_demand_da['datetime'] = pd.to_datetime(df_demand_da['datetime'])
    df_demand_da.set_index('datetime', inplace=True)
    df_demand_da.index = strip_tz(df_demand_da.index)
    df_dh = df_demand_da[['Demand_DA_MW']].resample('h').mean().rename(
        columns={'Demand_DA_MW': 'Demand_MW'})
    if (df_dh.index.max() - df_dh.index.min()).days < 30:
        HAS_DA_DEMAND = False
    else:
        df_demand_hourly = df_dh

if not HAS_DA_DEMAND:
    df_demand['Datetime'] = pd.to_datetime(df_demand['SETTLEMENT_DATE'],
        dayfirst=True, format='mixed') + \
        pd.to_timedelta((df_demand['SETTLEMENT_PERIOD']-1)*30, unit='m')
    df_demand.set_index('Datetime', inplace=True)
    df_demand_hourly = df_demand[['ND']].resample('h').mean().rename(
        columns={'ND': 'Demand_MW'}).shift(24).dropna()
    DEMAND_IS_PERSISTENCE = True

for df, col, out in [
    (df_wind_p,  'timestamp', 'Wind_Predicted_MW'),
    (df_solar_p, 'timestamp', 'Solar_Predicted_MW'),
]:
    df[col] = pd.to_datetime(df[col])
    df.set_index(df[col].dt.tz_convert(None), inplace=True)
df_wind_hourly  = df_wind_p[['predicted_mw']].resample('h').mean().rename(
    columns={'predicted_mw': 'Wind_Predicted_MW'})
df_solar_hourly = df_solar_p[['predicted_mw']].resample('h').mean().rename(
    columns={'predicted_mw': 'Solar_Predicted_MW'})

if HAS_DA_WINDSOLAR:
    df_ws_da['datetime'] = pd.to_datetime(df_ws_da['datetime'])
    df_ws_da.set_index('datetime', inplace=True)
    df_ws_da.index = strip_tz(df_ws_da.index)
    ws_da_cols = [c for c in df_ws_da.columns if 'Forecast' in c]
    df_ws_da_hourly = df_ws_da[ws_da_cols].resample('h').mean()
    if (df_ws_da_hourly.index.max() - df_ws_da_hourly.index.min()).days < 30:
        HAS_DA_WINDSOLAR = False; df_ws_da_hourly = pd.DataFrame()
else:
    df_ws_da_hourly = pd.DataFrame()

df_prices['Datetime'] = pd.to_datetime(df_prices['Datetime (UTC)'], dayfirst=True, format='mixed')
df_prices.set_index('Datetime', inplace=True)
df_prices_hourly = df_prices[['Price (EUR/MWhe)']].sort_index().rename(
    columns={'Price (EUR/MWhe)': 'Price_EUR'})

for df_daily, col_name in [(df_gas, 'Gas_Price'), (df_co2, 'CO2_Price')]:
    date_col = 'Date'
    price_col = ('SAP actual day' if 'SAP actual day' in df_daily.columns
                 else ('Price' if 'Price' in df_daily.columns else df_daily.columns[0]))
    df_daily[date_col] = pd.to_datetime(df_daily[date_col], dayfirst=True, format='mixed')
    df_daily.sort_values(date_col, inplace=True)
    df_daily.set_index(date_col, inplace=True)
    if col_name == 'Gas_Price':
        df_gas_daily2 = df_daily[[price_col]].rename(columns={price_col: 'Gas_Price'}).shift(1)
        df_gas_hourly = df_gas_daily2.reindex(
            pd.date_range(df_daily.index.min(), df_daily.index.max(), freq='D')
        ).interpolate().resample('h').ffill()
    else:
        df_co2_daily2 = df_daily[[price_col]].rename(columns={price_col: 'CO2_Price'}).shift(1)
        df_co2_hourly = df_co2_daily2.reindex(
            pd.date_range(df_daily.index.min(), df_daily.index.max(), freq='D')
        ).interpolate().resample('h').ffill()

data = df_prices_hourly.join(df_demand_hourly, how='inner')
data = data.join(df_wind_hourly, how='inner')
data = data.join(df_solar_hourly, how='inner')
data = data.join(df_gas_hourly, how='inner')
data = data.join(df_co2_hourly, how='inner')
if HAS_DA_WINDSOLAR and not df_ws_da_hourly.empty:
    data = data.join(df_ws_da_hourly, how='left')

data['Residual_Load']        = data['Demand_MW'] - (data['Wind_Predicted_MW'] + data['Solar_Predicted_MW'])
data['Theoretical_Cost']     = data['Gas_Price'] + (0.5 * data['CO2_Price'])
data['Cost_Load_Interaction'] = data['Theoretical_Cost'] * data['Residual_Load']
data['ResLoad_Roll_Mean_24']  = data['Residual_Load'].rolling(24, closed='left').mean()
data['ResLoad_Roll_Std_24']   = data['Residual_Load'].rolling(24, closed='left').std()

uk_hols = holidays.UnitedKingdom(years=list(range(2021, 2026)))
data['is_holiday'] = data.index.map(lambda x: 1 if x in uk_hols else 0)
data['hour_sin']   = np.sin(2 * np.pi * data.index.hour / 24)
data['hour_cos']   = np.cos(2 * np.pi * data.index.hour / 24)
data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
data['Price_Lag_24']       = data['Price_EUR'].shift(24)
data['Price_Lag_168']      = data['Price_EUR'].shift(168)
data['Price_Roll_Mean_24'] = data['Price_EUR'].rolling(24, closed='left').mean()

if 'Wind_Forecast_DA_MW' in data.columns:
    data['Wind_Forecast_Error_XGB'] = data['Wind_Predicted_MW'] - data['Wind_Forecast_DA_MW']
if 'Solar_Forecast_DA_MW' in data.columns:
    data['Solar_Forecast_Error_XGB'] = data['Solar_Predicted_MW'] - data['Solar_Forecast_DA_MW']
if 'Wind_Forecast_DA_MW' in data.columns and 'Solar_Forecast_DA_MW' in data.columns:
    data['Residual_Load_DA'] = data['Demand_MW'] - (data['Wind_Forecast_DA_MW'] + data['Solar_Forecast_DA_MW'])

data.dropna(inplace=True)

feature_cols = [
    'Residual_Load', 'ResLoad_Roll_Mean_24', 'ResLoad_Roll_Std_24',
    'Gas_Price', 'CO2_Price', 'Theoretical_Cost', 'Cost_Load_Interaction',
    'Wind_Predicted_MW', 'Solar_Predicted_MW', 'Demand_MW',
    'Price_Lag_24', 'Price_Lag_168', 'Price_Roll_Mean_24',
    'hour_sin', 'hour_cos', 'is_holiday', 'is_weekend'
]
for f in ['Wind_Forecast_DA_MW','Solar_Forecast_DA_MW',
          'Wind_Forecast_Error_XGB','Solar_Forecast_Error_XGB','Residual_Load_DA']:
    if f in data.columns:
        feature_cols.append(f)

train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]; test_data = data.iloc[train_size:]
X_train = train_data[feature_cols]; y_train = train_data['Price_EUR']
X_test  = test_data[feature_cols];  y_test  = test_data['Price_EUR']

print(f"Train: {len(train_data):,} ({train_data.index.min().date()} → {train_data.index.max().date()})")
print(f"Test : {len(test_data):,}  ({test_data.index.min().date()} → {test_data.index.max().date()})")
print(f"Features: {len(feature_cols)}")

scaler_X = StandardScaler(); scaler_y = StandardScaler()
X_tr_sc = scaler_X.fit_transform(X_train); X_te_sc = scaler_X.transform(X_test)
y_tr_sc = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()

def create_sequences(X, y, w):
    n = len(X) - w
    idx = np.arange(w)[None,:] + np.arange(n)[:,None]
    return X[idx], y[w:]

X_all, y_all = create_sequences(X_tr_sc, y_tr_sc, WINDOW)
Xc = np.vstack([X_tr_sc[-WINDOW:], X_te_sc])
yc = np.concatenate([y_tr_sc[-WINDOW:],
    scaler_y.transform(y_test.values.reshape(-1,1)).ravel()])
X_seq_test, _ = create_sequences(Xc, yc, WINDOW)

vs = int(len(X_all)*0.1)
Xtr, ytr = X_all[:-vs], y_all[:-vs]
Xval, yval = X_all[-vs:], y_all[-vs:]

Xtr_t  = torch.FloatTensor(Xtr);  ytr_t  = torch.FloatTensor(ytr)
Xval_t = torch.FloatTensor(Xval); yval_t = torch.FloatTensor(yval)
Xte_t  = torch.FloatTensor(X_seq_test)
n_feat = X_tr_sc.shape[1]

train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f"\nTrain seq={Xtr.shape}  Val seq={Xval.shape}  Test seq={X_seq_test.shape}")
print(f"y_std (train) = {scaler_y.scale_[0]:.2f} EUR/MWh")
y_std = float(scaler_y.scale_[0])

# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, H):
        scores  = self.v(torch.tanh(self.W(H)))
        weights = torch.softmax(scores, dim=1)
        return (weights * H).sum(dim=1)


def _init_forget_bias(lstm_module):
    """Set forget-gate bias to +1.0 — standard best practice for LSTM."""
    for name, param in lstm_module.named_parameters():
        if 'bias' in name:
            n = param.size(0)
            param.data[n//4:n//2].fill_(1.0)


class LSTMModel(nn.Module):
    """3-layer LSTM with temporal attention — PyTorch default init + forget bias=1."""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        _init_forget_bias(self.lstm)
        self.attn = TemporalAttention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        H, _ = self.lstm(x)
        return self.head(self.attn(H)).squeeze(-1)


class BiLSTMModel(nn.Module):
    """3-layer BiLSTM with temporal attention — PyTorch default init + forget bias=1."""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout, bidirectional=True)
        _init_forget_bias(self.bilstm)
        self.attn = TemporalAttention(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        H, _ = self.bilstm(x)
        return self.head(self.attn(H)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
_DL_CONFIGS_V2 = {
    'LSTM v2':   {'max_lr': 3e-4, 'weight_decay': 1e-2, 'grad_clip': 1.0},
    'BiLSTM v2': {'max_lr': 3e-4, 'weight_decay': 1e-2, 'grad_clip': 1.0},
}


def train(model, name):
    cfg = _DL_CONFIGS_V2[name]
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg['max_lr']/25, weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg['max_lr'],
        epochs=MAX_EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=WARMUP_PCT, anneal_strategy='cos')
    crit = nn.MSELoss()

    best_val = float('inf'); best_state = None; wait = 0
    Xv, yv = Xval_t.to(DEVICE), yval_t.to(DEVICE)
    tloss, vloss = [], []

    print(f"  {name}: max_lr={cfg['max_lr']}, wd={cfg['weight_decay']}, "
          f"grad_clip={cfg['grad_clip']}")

    for epoch in range(MAX_EPOCHS):
        model.train(); rl = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step(); sched.step()
            rl += loss.item()
        avg_t = rl / len(train_loader)
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        tloss.append(avg_t); vloss.append(vl)

        training_logger.log_epoch(name, epoch+1, MAX_EPOCHS,
            train_loss=avg_t, val_loss=vl,
            lr=opt.param_groups[0]['lr'],
            patience_counter=wait, best_val_loss=best_val)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  {name}: Early stop @ epoch {epoch+1}  "
                      f"(best val MSE: {best_val:.6f})")
                break
        if (epoch+1) % 50 == 0:
            print(f"  {name}: Epoch {epoch+1}/{MAX_EPOCHS}  "
                  f"val={vl:.5f}  lr={opt.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state); model.eval()
    best_ep = len(tloss) - wait
    return model, tloss, vloss, best_ep


# ──────────────────────────────────────────────────────────────────────────────
training_logger.log_stage("LSTM/BiLSTM Retrain v2 (max_lr=3e-4, forget_bias=1)")

dl_models = {
    'LSTM v2':   LSTMModel(n_feat),
    'BiLSTM v2': BiLSTMModel(n_feat),
}

results = []
for name, model in dl_models.items():
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams  = {**_DL_CONFIGS_V2[name],
                'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
                'batch_size': BATCH_SIZE, 'window': WINDOW,
                'n_params': n_params, 'note': 'forget_bias=1, PyTorch default init'}
    training_logger.log_model_start(name, model_type='pytorch',
                                    hyperparams=hparams, category='Deep Learning')
    t0 = time.time()
    print(f"\n{'='*60}\nTraining {name}  ({n_params:,} params)\n{'='*60}")

    model, tloss, vloss, best_ep = train(model, name)

    preds_sc = []
    with torch.no_grad():
        for i in range(0, len(Xte_t), BATCH_SIZE):
            preds_sc.append(model(Xte_t[i:i+BATCH_SIZE].to(DEVICE)).cpu())
    preds_sc = torch.cat(preds_sc).numpy()
    preds    = scaler_y.inverse_transform(preds_sc.reshape(-1,1)).ravel()

    r2   = r2_score(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    dur  = time.time() - t0

    tr_rmse = math.sqrt(tloss[best_ep-1]) * y_std
    vl_rmse = math.sqrt(vloss[best_ep-1]) * y_std

    row = {'Model': name, 'R2': round(r2,4), 'RMSE': round(rmse,2),
           'MAE': round(mae,2), 'Train_RMSE': round(tr_rmse,2),
           'Val_RMSE': round(vl_rmse,2), 'Best_Epoch': best_ep}
    results.append(row)

    training_logger.log_model_done(name, r2=r2, rmse=rmse, mae=mae,
                                   duration_s=dur, category='Deep Learning')
    print(f"\n  {name}: R2={r2:.4f}  Test RMSE={rmse:.2f}  MAE={mae:.2f}")
    print(f"  Train RMSE={tr_rmse:.2f}  Val RMSE={vl_rmse:.2f}  "
          f"Best epoch={best_ep}  Duration={dur:.0f}s")

# ──────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("  LSTM/BiLSTM RETRAIN v2 RESULTS")
print("="*70)
for row in results:
    print(f"  {row['Model']:<12}  R2={row['R2']:.4f}  "
          f"RMSE={row['RMSE']:.2f}  MAE={row['MAE']:.2f}  "
          f"TrainRMSE={row['Train_RMSE']:.2f}  ValRMSE={row['Val_RMSE']:.2f}  "
          f"BestEpoch={row['Best_Epoch']}")

print("\nFor reference:")
print("  DNN      R2=0.8173  RMSE=15.14  MAE=10.99")
print("  TCN      R2=0.8514  RMSE=13.86  MAE=9.96")
print("  PatchTST R2=0.6742  RMSE=20.52  MAE=14.80")
print("="*70)
