# NPCNet: Navigator-Driven Pseudo Text for Deep Clustering of Early Sepsis Phenotyping

## Data Preparation
All csv files should be placed in the same directory as `main.py`.
### 1. texts.csv
This file contains pseudo texts (`text`) that preserve the chronological order of clinical examinations, along with the corresponding tokenized inputs (`token` and `mask`) required by NPCNet.  
In the example shown below, the maximum number of examinations for all patients is four, and the sequences are padded accordingly to ensure all sequences have the same maximum length across patients. `mask` ensures padded positions (0s) are ignored during the training stage.  
Example:
```
subject_id| event|                                                                      text|                          token|                mask
     12345|     1| glucose glucose-6 FiO2 FiO2-9 lactate lactate-8 bicarbonate bicarbonate-9| 1 9 309 23 155 26 193 25 184 2| 1 1 1 1 1 1 1 1 1 1
     12457|     1|         lactate lactate-8 chloride chloride-4 glucose glucose-1 BUN BUN-4| 1 26 193 10 297 9 304 11 209 2| 1 1 1 1 1 1 1 1 1 1
     12457|     2|                                         hemoglobin hemoglobin-4 BUN BUN-9|      1 31 258 11 214 2 0 0 0 0| 1 1 1 1 1 1 0 0 0 0
     12548|     1|                                  glucose glucose-10 FiO2 FiO2-2 pO2 pO2-2|    1 9 313 23 149 22 139 2 0 0| 1 1 1 1 1 1 1 1 0 0
     12548|     2|                       bands bands-10 lymphocytes lymphocytes-10 WBC WBC-1|       1 4 56 17 88 18 89 2 0 0| 1 1 1 1 1 1 1 1 0 0
```
### 2. single_points.csv
This file includes the variables listed in `static_var_cols` and the target navigator in the `main.py`.  
Example:
```
subject_id| event| gender| anchor_age| comorbidity_1| ...| comorbidity_n| target
     12345|     1|      M|         68|             1| ...|             0|      1
     12457|     1|      M|         80|             0| ...|             0|      1
     12457|     2|      F|         34|             0| ...|             0|      0
     12548|     1|      M|         57|             1| ...|             1|      0
     12548|     2|      F|         63|             1| ...|             0|      0
```
## Getting Started
### 1. Clone the repository:
```
git clone https://github.com/DHLab-TSENG/NPCNet.git
```
### 2. Change directory to the project directory:
```
cd NPCNet
```
### 3. Install dependencies:
```
pip install -r requirements.txt
```
### 4. Run the model:
After placing `texts.csv` and `single_points.csv` in the same directory as main.py, run the following command:
```
python main.py
```
### 5. Get the output.csv
This file outputs the deep representations of the patients and their cluster assignments.  
Example:
```
subject_id| event| embedding_1| ...| embedding_n| cluster
     12345|     1|       0.007| ...|      -0.158|       2
     12457|     1|       0.218| ...|      -0.232|       0
     12457|     2|       0.148| ...|      -0.205|       1
     12548|     1|       0.169| ...|      -0.222|       2
     12548|     2|       0.148| ...|      -0.208|       3
```
