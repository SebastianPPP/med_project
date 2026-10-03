import os
import json
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
import optuna
import joblib
import shap
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from lifelines.utils import concordance_index

# Wyłączenie logów Optuny
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. ŁADOWANIE I PRZYGOTOWANIE DANYCH
# ==========================================
print("Ładowanie danych...")
df = pd.read_csv('SEER Breast Cancer Dataset Cleaned.csv')
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

df['event'] = (df['status'].str.lower() == 'dead').astype(int)
df['time'] = df['survival_months'].astype(float)

X = df.drop(columns=['survival_months', 'status', 'event', 'time'])
X_encoded = pd.get_dummies(X, drop_first=True)

# ==========================================
# 2. ZAAWANSOWANA INŻYNIERIA CECH (POLYNOMIAL)
# ==========================================
print("Generowanie interakcji i potęg cech...")
num_cols = ['age', 'tumor_size', 'regional_node_examined', 'reginol_node_positive']
X_num = X_encoded[[col for col in num_cols if col in X_encoded.columns]]
X_cat = X_encoded.drop(columns=X_num.columns)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_num_poly = pd.DataFrame(
    poly.fit_transform(X_num), 
    columns=poly.get_feature_names_out(X_num.columns)
)

X_full = pd.concat([X_num_poly.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

# Selekcja na bazie korelacji z czasem
correlations = X_full.corrwith(df['time']).abs()
top_features = correlations[correlations > 0.03].index.tolist()
print(f"Liczba cech po wygenerowaniu interakcji i selekcji: {len(top_features)}")

df_final = X_full[top_features].copy()
df_final['time'] = df['time']
df_final['event'] = df['event']

# Podział na zbiór treningowy, walidacyjny i testowy
df_temp, df_test = train_test_split(df_final, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_temp, test_size=0.2, random_state=42)

scaler = StandardScaler()
df_train[top_features] = scaler.fit_transform(df_train[top_features])
df_val[top_features] = scaler.transform(df_val[top_features])
df_test[top_features] = scaler.transform(df_test[top_features])

# ==========================================
# 3. OPTYMALIZACJA BAYESOWSKA (OPTUNA)
# ==========================================
print("\nRozpoczynam optymalizację bayesowską za pomocą Optuny (30 prób)...")

y_train_xgb = np.where(df_train['event'] == 1, df_train['time'], -df_train['time'])
y_val_xgb = np.where(df_val['event'] == 1, df_val['time'], -df_val['time'])
y_test_xgb = np.where(df_test['event'] == 1, df_test['time'], -df_test['time'])

dtrain = xgb.DMatrix(df_train[top_features], label=y_train_xgb)
dval = xgb.DMatrix(df_val[top_features], label=y_val_xgb)
dtest = xgb.DMatrix(df_test[top_features], label=y_test_xgb)

def objective(trial):
    params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'eta': trial.suggest_float('eta', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        'seed': 42
    }
    
    tmp_model = xgb.train(params, dtrain, num_boost_round=100)
    tmp_preds = tmp_model.predict(dval)
    c_index = concordance_index(df_val['time'], -tmp_preds, df_val['event'])
    return c_index

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

best_params = study.best_params
best_params['objective'] = 'survival:cox'
best_params['eval_metric'] = 'cox-nloglik'
best_params['seed'] = 42

print(f"Optymalny C-index na walidacji: {study.best_value:.4f}")

# ==========================================
# 4. TRENOWANIE FINALNEGO MODELU ORAZ SHAP
# ==========================================
print("\nTrenowanie ostatecznego modelu oraz konfiguracja wyjaśniacza SHAP...")
df_train_full = pd.concat([df_train, df_val])
y_train_full_xgb = np.where(df_train_full['event'] == 1, df_train_full['time'], -df_train_full['time'])
dtrain_full = xgb.DMatrix(df_train_full[top_features], label=y_train_full_xgb)

final_xgb_model = xgb.train(best_params, dtrain_full, num_boost_round=150)

# Ewaluacja ostateczna
risk_xgb = final_xgb_model.predict(dtest)
test_c_index_xgb = concordance_index(df_test['time'], -risk_xgb, df_test['event'])
print(f"--> FINAL XGBoost_Survival | Test C-index: {test_c_index_xgb:.4f}")

# Zapisywanie artefaktów i tła SHAP
os.makedirs('metrics', exist_ok=True)
final_xgb_model.save_model('metrics/xgb_survival_model.json')
joblib.dump(scaler, 'metrics/scaler.pkl')
joblib.dump(poly, 'metrics/poly.pkl')
joblib.dump(top_features, 'metrics/top_features.pkl')
joblib.dump(num_cols, 'metrics/num_cols.pkl')
joblib.dump(X_cat.columns.tolist(), 'metrics/cat_cols.pkl')

# SHAP Explainer
background_data = df_train[top_features].iloc[:100]
explainer = shap.TreeExplainer(final_xgb_model, data=background_data)
joblib.dump(explainer, 'metrics/shap_explainer.pkl')

metrics = {
    'Final_XGBoost_Survival_C_Index': round(test_c_index_xgb, 4),
    'Validation_C_Index': round(study.best_value, 4)
}
with open(os.path.join('metrics', 'survival_metrics_optuna.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n[SUKCES] Model, artefakty i explainer SHAP zostały pomyślnie zapisane!")