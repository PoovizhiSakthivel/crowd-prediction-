import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import joblib, os, warnings
warnings.filterwarnings('ignore')

# ── Load dataset ───────────────────────────────────────────────────
df = pd.read_csv("Historical_crowd_dataset.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Feature Engineering ────────────────────────────────────────────
df['Hour']        = df['Time'].apply(lambda t: int(str(t).split(':')[0]))
df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month']       = df['Date_parsed'].dt.month.fillna(1).astype(int)
df['DayOfMonth']  = df['Date_parsed'].dt.day.fillna(1).astype(int)

day_order = {'Monday':0,'Tuesday':1,'Wednesday':2,
             'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
df['DayNum']       = df['Day'].map(day_order).fillna(0).astype(int)
df['Is_Weekend']   = df['DayNum'].apply(lambda x: 1 if x >= 5 else 0)
df['Is_Peak_Hour'] = df['Hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19,20] else 0)
df['Is_Saturday']  = (df['DayNum'] == 5).astype(int)
df['Is_Sunday']    = (df['DayNum'] == 6).astype(int)
df['Is_Rainy']     = (df['Weather'] == 'Rainy').astype(int)
df['Hot_Day']      = (df['Temperature'] > 33).astype(int)

# Cyclical encoding
df['Hour_sin']  = np.sin(2*np.pi*df['Hour']/24)
df['Hour_cos']  = np.cos(2*np.pi*df['Hour']/24)
df['Month_sin'] = np.sin(2*np.pi*df['Month']/12)
df['Month_cos'] = np.cos(2*np.pi*df['Month']/12)
df['Day_sin']   = np.sin(2*np.pi*df['DayNum']/7)
df['Day_cos']   = np.cos(2*np.pi*df['DayNum']/7)

# Interaction features
df['Weekend_Peak']  = df['Is_Weekend'] * df['Is_Peak_Hour']
df['Weekend_Rainy'] = df['Is_Weekend'] * df['Is_Rainy']
df['Sat_Peak']      = df['Is_Saturday'] * df['Is_Peak_Hour']
df['Sun_Peak']      = df['Is_Sunday']   * df['Is_Peak_Hour']

# ── Encode categoricals ────────────────────────────────────────────
encoders = {}
for col in ['Location', 'Destination', 'Weather']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── Target ─────────────────────────────────────────────────────────
crowd_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
df['Crowd_Label'] = df['Crowd_Level'].map(crowd_map).fillna(0).astype(int)

# ── Features ───────────────────────────────────────────────────────
FEATURES = [
    'Hour', 'DayNum', 'Month', 'DayOfMonth',
    'Location_enc', 'Destination_enc', 'Weather_enc', 'Temperature',
    'Is_Weekend', 'Is_Peak_Hour', 'Is_Saturday', 'Is_Sunday',
    'Is_Rainy', 'Hot_Day',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
    'Weekend_Peak', 'Weekend_Rainy', 'Sat_Peak', 'Sun_Peak'
]

X = df[FEATURES]
y = df['Crowd_Label']

# ── Stratified split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Balance classes — 1200 per class ──────────────────────────────
train_df = X_train.copy()
train_df['label'] = y_train.values
parts = []
for cls in [0, 1, 2, 3]:
    cls_df = train_df[train_df['label'] == cls]
    cls_df = resample(cls_df, replace=(len(cls_df) < 1200),
                      n_samples=1200, random_state=42)
    parts.append(cls_df)
balanced = pd.concat(parts).sample(frac=1, random_state=42)
X_res = balanced[FEATURES]
y_res = balanced['label']
print(f"Balanced class counts: { {k: (y_res==k).sum() for k in [0,1,2,3]} }")

# ── Train RandomForestClassifier (Good Fit parameters) ────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,                       # shallow → less memorization
    min_samples_leaf=20,               # strict leaf → generalization
    min_samples_split=40,              # strict split → simpler trees
    max_features='log2',               # fewer features/split → diversity
    max_samples=0.7,                   # 70% row subsampling → diversity
    class_weight='balanced_subsample',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
model.fit(X_res, y_res)

# ── Evaluate ───────────────────────────────────────────────────────
y_pred    = model.predict(X_test)
train_acc = model.score(X_train, y_train) * 100
test_acc  = accuracy_score(y_test, y_pred) * 100
oob_acc   = model.oob_score_ * 100
gap       = train_acc - test_acc

print("\n── Model Evaluation ──────────────────────────────────")
print(f"Training Accuracy  : {train_acc:.2f}%")
print(f"OOB Accuracy       : {oob_acc:.2f}%")
print(f"Test Accuracy      : {test_acc:.2f}%")
print(f"Gap (Train-Test)   : {gap:.2f}%")

if gap < 5:    verdict = "✅  Good Fit"
elif gap < 15: verdict = "⚠️   Slight Overfit"
else:          verdict = "❌  Overfit"
print(f"Verdict            : {verdict}")

print("\n── Per-Class Report ──────────────────────────────────")
labels = ['Low', 'Medium', 'High', 'Very High']
print(classification_report(y_test, y_pred, target_names=labels))

print("\n── Cross Validation (5-fold) ─────────────────────────")
cv = cross_val_score(model, X_res, y_res, cv=5)
print(f"CV Scores : {[round(s*100,2) for s in cv]}")
print(f"CV Mean   : {cv.mean()*100:.2f}%  |  CV Std : {cv.std()*100:.2f}%")

print("\n── Feature Importances ───────────────────────────────")
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(fi.to_string())

# ── Save artifacts ─────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,    "model/crowd_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")
meta = {
    "locations":    sorted(df["Location"].unique().tolist()),
    "destinations": sorted(df["Destination"].unique().tolist()),
    "weathers":     sorted(df["Weather"].unique().tolist()),
    "crowd_labels": {v: k for k, v in crowd_map.items()},
    "features":     FEATURES,
}
joblib.dump(meta, "model/meta.pkl")
print("\n✅  Saved: model/crowd_model.pkl | model/encoders.pkl | model/meta.pkl")