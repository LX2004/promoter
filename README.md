# promoter
The supplementary folder contains the supplemental materials for the paper.
The generated promoter sequences and models are located in the folders 'sequences' and 'model', respectively.
# Preparation
run:

```
cd promoter

conda create --name promoter

conda activate promoter

conda install --file request.txt
```

# If you want to train ddpm for generating E. coli/cyanobacteria promoters, you can perform the following actions.
run :
```
cd train_generate_E_coli_promoter or train_generate_E_coli_promotercyanobacteria

python train_generate_model.py

python generate_promoters
```
# For prediction task
run : 
```
cd train_prediction_model

python make_dataset.py

python prediction_transformer_dimer_original_kfold.py

python test_model_performance.py
```
