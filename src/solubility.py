import pandas as pd
import numpy as np
from pathlib import Path
import torch
from lightning import pytorch as pl

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from chemprop import data, featurizers, utils
from chemprop.models import multi_prod

test_path = "../data/dataset_sol.csv"
smiles_columns = ["SMILES 1", "SMILES 2"]
target_columns = ["Pressure EXP (bar)"]
features_columns = ["Liquid Fraction 1 (mol/mol)", "Temperature (°C)", "Liquid Fraction 2 (mol/mol)"]
mole_fraction_columns = ["Liquid Fraction 1 (mol/mol)", "Liquid Fraction 2 (mol/mol)"]

df_test = pd.read_csv(test_path)
df_test["SampleID"] = df_test.index

smiss = df_test.loc[:, smiles_columns].values
ys = df_test.loc[:, target_columns].values

extra_mol_features = df_test.loc[:, features_columns]
extra_mol_features[mole_fraction_columns] = extra_mol_features[mole_fraction_columns] * 0.85
extra_mol_features['Temperature (°C)'] = (extra_mol_features['Temperature (°C)'] + 273.15) / 373.15 * 0.3
for column in mole_fraction_columns:
    extra_mol_features[column] = extra_mol_features[column] * extra_mol_features['Temperature (°C)']

extra_mol_features = extra_mol_features.drop(columns=["Temperature (°C)"])
extra_features_descriptors = extra_mol_features.values

test_datapointss = []

solute_data = [
    data.MoleculeDatapoint(
        mol=utils.make_mol(smiss[i, 0], keep_h=False, add_h=False),
        x_d=extra_features_descriptors[i]
    )
    for i in range(len(smiss))
]
test_datapointss.append(solute_data)

for col_idx in range(1, len(smiles_columns)):
    solvent_data = [
        data.MoleculeDatapoint(
            mol=utils.make_mol(smiss[i, col_idx], keep_h=False, add_h=False),
            x_d=extra_features_descriptors[i]
        )
        for i in range(len(smiss))
    ]
    test_datapointss.append(solvent_data)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
test_dsets = [data.MoleculeDataset(test_datapoints, featurizer) for test_datapoints in test_datapointss]

test_mcdset = data.MulticomponentDataset(test_dsets)
test_loader = data.build_dataloader(test_mcdset, shuffle=False)

n_splits = 3
n_repeats = 3
random_state = 42

num_folds = n_splits*n_repeats
ckpt_dir = Path("../checkpoints")

all_test_results = []
fold_metrics = []

for fold_idx in range(num_folds):
    rep = fold_idx // n_splits + 1
    fold = fold_idx % n_splits + 1

    print(f"\nfold = {fold}, rep = {rep}\n")

    pattern = f"fine-fold={fold}-rep={rep}-epoch=*.ckpt"
    matches = list(ckpt_dir.glob(pattern))
    if not matches:
        print(f"Checkpoint non trovato per fold={fold}, rep={rep}")
        continue
    
    checkpoint_path = matches[0]  

    mpnn_cls = multi_prod.MulticomponentMPNN_mod
    mcmpnn = mpnn_cls.load_from_checkpoint(checkpoint_path)

    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="cpu",
            devices=1
    )
    test_preds = trainer.predict(mcmpnn, test_loader)

    test_preds = np.concatenate(test_preds, axis=0)

    test_results_df = df_test.copy()
    test_results_df["Pressure PRED (bar)"] = test_preds
    test_results_df["Pressure EXP (bar)"] = ys

    test_results_df["Fold"] = fold
    test_results_df["Replicate"] = rep
    all_test_results.append(test_results_df)

    mae = mean_absolute_error(test_results_df['Pressure EXP (bar)'], test_results_df["Pressure PRED (bar)"])
    mse = mean_squared_error(test_results_df['Pressure EXP (bar)'], test_results_df["Pressure PRED (bar)"])
    r2 = r2_score(test_results_df['Pressure EXP (bar)'], test_results_df["Pressure PRED (bar)"])

    print(f"Fold {fold}, Rep {rep} - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")

    fold_metrics.append({
        "Fold": fold,
        "Replicate": rep,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    })

final_test_results = pd.concat(all_test_results, ignore_index=True)

metrics_df = pd.DataFrame(fold_metrics)

final_agg = (
    final_test_results
    .groupby("SampleID")
    .agg({
        "Component 1": "first",
        "Component 2": "first",
        "Liquid Fraction 1 (mol/mol)": "first",
        "Liquid Fraction 2 (mol/mol)": "first",
        "Temperature (°C)": "first",
        "Pressure EXP (bar)": "mean",
        "Pressure PRED (bar)": ["mean", "std"],
    })
)

final_agg.columns = [
    "Component 1",
    "Component 2",
    "Liquid Fraction 1 (mol/mol)",
    "Liquid Fraction 2 (mol/mol)",
    "Temperature (°C)",
    "OriginalValue",        # Pressure EXP (bar)
    "MeanPrediction",       # Pressure PRED (bar), mean
    "StdPrediction",        # Pressure PRED (bar), std
]

final_agg = final_agg.reset_index() 
final_agg_output_path = "../results/aggregated_predictions_sol.csv"
final_agg.to_csv(final_agg_output_path, index=False)

print(f"Final aggregated predictions saved in: {final_agg_output_path}")

print(final_agg.head())
