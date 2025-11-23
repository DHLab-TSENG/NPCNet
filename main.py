import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import torch
from torch.utils.data import DataLoader
from utils import seed_everything, args
from dataset import SepsisDataset
from model import ClusteringModule, result

static_var_cols = ["gender",
                   "congestive_heart_failure",
                   "cardiac_arrhythmias",
                   "valvular_disease",
                   "pulmonary_circulation_disorders",
                   "peripheral_vascular_disorders",
                   "hypertension_uncomplicated",
                   "hypertension_complicated",
                   "paralysis",
                   "other_neurological_disorders",
                   "chronic_pulmonary_disease",
                   "diabetes_uncomplicated",
                   "diabetes_complicated",
                   "hypothyroidism",
                   "renal_failure",
                   "liver_disease",
                   "peptic_ulcer_disease_excluding_bleeding",
                   "AIDS_H1V",
                   "lymphoma",
                   "metastatic_cancer",
                   "solid_tumor_without_metastasis",
                   "rheumatoid_arthritis_collagen_vascular_diseases",
                   "coagulopathy",
                   "obesity",
                   "weight_loss",
                   "fluid_and_electrolyte_disorders",
                   "blood_loss_anemia",
                   "deficiency_anemia",
                   "alcohol_abuse",
                   "drug_abuse",
                   "psychoses",
                   "depression",
                   "age"]

def main():
    seed_everything(args["seed"])

    texts = pd.read_csv("texts.csv")
    all_inputs = texts["token"].apply(lambda x: torch.tensor(list(map(int, x.split(" ")))))
    all_inputs = torch.stack(all_inputs.tolist())
    all_masks = texts["mask"].apply(lambda x: torch.tensor(list(map(int, x.split(" ")))))
    all_masks = torch.stack(all_masks.tolist())
    
    single_points = pd.read_csv("single_points.csv")
    subject_idx = single_points["subject_id"].unique()

    training_set = np.random.choice(subject_idx, size = int(0.8 * len(subject_idx)), replace = False)
    training_idx = single_points.index[single_points["subject_id"].isin(training_set)]
    val_idx = single_points.index[~single_points["subject_id"].isin(training_set)]

    training_inputs, val_inputs = all_inputs[training_idx], all_inputs[val_idx]
    training_masks, val_masks = all_masks[training_idx].float(), all_masks[val_idx].float()
    
    single_points.loc[single_points["gender"] == "F", "gender"] = 0
    single_points.loc[single_points["gender"] == "M", "gender"] = 1
    single_points["age"] = pd.cut(single_points["anchor_age"], bins = 10, labels = False)

    statics = torch.tensor(single_points[static_var_cols].values.tolist())
    training_statics, val_statics = statics[training_idx], statics[val_idx]
    
    ys = single_points["hospital_expire_flag"]
    training_ys, val_ys = ys.loc[training_idx].reset_index(drop = True), ys.loc[val_idx].reset_index(drop = True)

    train_loader = DataLoader(SepsisDataset(training_inputs, training_masks, training_statics, training_ys), batch_size = args["batch_size"], shuffle = True)

    model = ClusteringModule(args)
    model.pretrain(args["pretraining_epoch"], train_loader)
    for epoch in range(args["epoch"]):
        model.train()
        cluster_centroids = model.fit(epoch, train_loader)
        
    training_output = result(training_inputs, training_masks, training_statics, training_idx, single_points, model, cluster_centroids)
    val_output = result(val_inputs, val_masks, val_statics, val_idx, single_points, model, cluster_centroids)
    
    embedding = [f"embedding{i}" for i in range(1, 33)]
    SI = silhouette_score(np.array(val_output[embedding]), np.array(val_output["cluster"]))
    CHI = calinski_harabasz_score(np.array(val_output[embedding]), np.array(val_output["cluster"]))
    DBI = davies_bouldin_score(np.array(val_output[embedding]), np.array(val_output["cluster"]))
    print("SI:", round(SI, 3))
    print("CHI:", round(CHI / len(val_output), 3))
    print("DBI:", round(DBI, 3))
    
    pd.concat([training_output, val_output], ignore_index = True).to_csv("output.csv", sep = ',', index = False, header = True)

if __name__ == "__main__":
    main()