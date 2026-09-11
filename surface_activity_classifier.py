import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

MODEL = "tabpfn"  # "random_forest" or "tabpfn"
CONCENTRATION = 0.5

df = pd.read_csv("Datasets/c16_surf_clean.csv", sep = ';')
df = df[df["concentration"] == CONCENTRATION].reset_index(drop=True)
surface_tension = df["surface_tension"].values

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
pH_scaled = StandardScaler().fit_transform(df[["pH"]].values)
sidechain_scaled = StandardScaler().fit_transform(df[["net_charge_sidechain"]].values)

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
kept_descriptor_names = [n for n, drop in zip(all_descriptor_names, bad_cols) if not drop]

train_idx, holdout_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
desc_scaler = StandardScaler().fit(X_full_desc_clean[train_idx])
X_full_desc_scaled = desc_scaler.transform(X_full_desc_clean)

lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_full_desc_scaled[train_idx], surface_tension[train_idx])
selected_idx = np.where(lasso.coef_ != 0)[0]
X_desc_optimized = X_full_desc_scaled[:, selected_idx]

X = np.hstack([X_desc_optimized, pH_scaled, sidechain_scaled])
y = (surface_tension < 50).astype(int)

if MODEL == "random_forest":
    clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
elif MODEL == "tabpfn":
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(random_state=42)
else:
    raise ValueError(f"unknown MODEL: {MODEL}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracy = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
cv_auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")

oof_pred = cross_val_predict(clf, X, y, cv=skf, method="predict")
oof_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]

print(f"model: {MODEL}")
print(f"CV accuracy: {cv_accuracy.mean():.3f} +/- {cv_accuracy.std():.3f}")
print(f"CV ROC-AUC:  {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
print(f"overall out-of-fold accuracy: {accuracy_score(y, oof_pred):.3f}")
print(f"overall out-of-fold ROC-AUC:  {roc_auc_score(y, oof_proba):.3f}")
print()
print(classification_report(y, oof_pred, target_names=["inactive", "active"]))

cm = confusion_matrix(y, oof_pred)
print("confusion matrix:")
print("                predicted inactive  predicted active")
print(f"actual inactive        {cm[0,0]:6d}            {cm[0,1]:6d}")
print(f"actual active          {cm[1,0]:6d}            {cm[1,1]:6d}")
