import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

MODEL = "random_forest"  # "random_forest" or "tabpfn"
CONCENTRATION = 0.5

df = pd.read_csv("../Datasets/c16_surf_clean.csv", sep=";")
df = df[df["concentration"] == CONCENTRATION].reset_index(drop=True)
surface_tension = df["surface_tension"].values
groups = df["sequence"].values

PKA = {
    "D": (3.65, "acidic"), "E": (4.25, "acidic"), "H": (6.00, "basic"),
    "K": (10.53, "basic"), "R": (12.48, "basic"),
}

def net_charge_sidechain_only(sequence, pH):
    charge = 0.0
    for aa in sequence:
        if aa in PKA:
            pKa, kind = PKA[aa]
            charge += 1 / (1 + 10 ** (pH - pKa)) if kind == "basic" else -1 / (1 + 10 ** (pKa - pH))
    return charge

df["net_charge_sidechain"] = [net_charge_sidechain_only(seq, ph) for seq, ph in zip(df["sequence"], df["pH"])]
pH_raw = df[["pH"]].values
sidechain_raw = df[["net_charge_sidechain"]].values

all_descriptor_names = [name for name, func in Descriptors._descList]
descriptor_funcs_all = {name: func for name, func in Descriptors._descList}

def compute_all_descriptors(df, descriptor_names):
    rows = []
    for smi in df["smiles_PA"]:
        mol = Chem.MolFromSmiles(smi)
        row = []
        for name in descriptor_names:
            try:
                val = descriptor_funcs_all[name](mol)
            except Exception:
                val = np.nan
            row.append(val)
        rows.append(row)
    return np.array(rows, dtype=float)

X_full_desc = compute_all_descriptors(df, all_descriptor_names)
bad_cols = np.isnan(X_full_desc).any(axis=0) | (np.nanstd(X_full_desc, axis=0) < 1e-10)
X_full_desc_clean = X_full_desc[:, ~bad_cols]

y = (surface_tension < 50).astype(int)

if MODEL == "random_forest":
    def make_model():
        return RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
elif MODEL == "tabpfn":
    from tabpfn import TabPFNClassifier
    def make_model():
        return TabPFNClassifier(random_state=42)
else:
    raise ValueError(f"unknown MODEL: {MODEL}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_pred = np.zeros(len(y), dtype=int)
oof_proba = np.zeros(len(y))
fold_accuracy = []
fold_auc = []

for train_idx, test_idx in sgkf.split(X_full_desc_clean, y, groups):
    desc_scaler = StandardScaler().fit(X_full_desc_clean[train_idx])
    X_desc_scaled = desc_scaler.transform(X_full_desc_clean)

    lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_desc_scaled[train_idx], surface_tension[train_idx])
    selected_idx = np.where(lasso.coef_ != 0)[0]
    X_desc_selected = X_desc_scaled[:, selected_idx]

    ph_scaler = StandardScaler().fit(pH_raw[train_idx])
    sidechain_scaler = StandardScaler().fit(sidechain_raw[train_idx])
    pH_scaled = ph_scaler.transform(pH_raw)
    sidechain_scaled = sidechain_scaler.transform(sidechain_raw)

    X_fold = np.hstack([X_desc_selected, pH_scaled, sidechain_scaled])

    model = make_model()
    model.fit(X_fold[train_idx], y[train_idx])

    proba = model.predict_proba(X_fold[test_idx])[:, 1]
    pred = model.predict(X_fold[test_idx])

    oof_proba[test_idx] = proba
    oof_pred[test_idx] = pred
    fold_accuracy.append(accuracy_score(y[test_idx], pred))
    fold_auc.append(roc_auc_score(y[test_idx], proba))

fold_accuracy = np.array(fold_accuracy)
fold_auc = np.array(fold_auc)

print(f"model: {MODEL}")
print(f"CV accuracy: {fold_accuracy.mean():.3f} +/- {fold_accuracy.std():.3f}")
print(f"CV ROC-AUC:  {fold_auc.mean():.3f} +/- {fold_auc.std():.3f}")
print(f"overall out-of-fold accuracy: {accuracy_score(y, oof_pred):.3f}")
print(f"overall out-of-fold ROC-AUC:  {roc_auc_score(y, oof_proba):.3f}")
print()
print(classification_report(y, oof_pred, target_names=["inactive", "active"]))

cm = confusion_matrix(y, oof_pred)
print("confusion matrix:")
print("                predicted inactive  predicted active")
print(f"actual inactive        {cm[0,0]:6d}            {cm[0,1]:6d}")
print(f"actual active          {cm[1,0]:6d}            {cm[1,1]:6d}")