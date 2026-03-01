import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = {
    "seed": 42,
    "pretraining_epoch": 3,
    "epoch": 20,
    "batch_size": 64,
    "lr": 1e-4,
    "wd": 1e-7,
    "n_clusters": 4,
    "vocab_size": 406,
    "latent_dim": 32,
    "w": 0.7,
    "lambda1": 1,
    "lambda2": 1e-3,
    "lambda3": 1,
    "kappa1": 1,
    "kappa2": 1e-2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

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
static_var_catnums = [2 for i in range(32)] + [10]
