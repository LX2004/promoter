# promoter
# The supplementary folder contains the supplemental materials for the paper.
# The generated promoter sequences and models are located in the folders 'sequences' and 'model', respectively.
# if you want to train ddpm for generating E. coli/cyanobacteria promoter, you can perform the following actions.
run :
cd train_generate_E_coli_promoter/train_generate_E_coli_promotercyanobacteria
python train_generate_model.py
