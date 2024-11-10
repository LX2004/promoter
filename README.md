# promoter
The supplementary folder contains the supplemental materials for the paper.
The generated promoter sequences and models are located in the folders 'sequences' and 'model', respectively.
# Preparation
In the first step, you should create env for code. Follow the steps below to create a virtual environment named "promoter." All code for this project must be run within this environment. run:

```
cd promoter

conda env create -f environment.yml

conda activate promoter

```
If the above method does not allow for a quick installation, the user can create a virtual environment named "promoter." Then, open the "environment.yml" file to check the environment dependencies and manually install them using conda or pip. The specific steps are as follows:

```
cd promoter

conda create -n promoter python=3.10

conda activate promoter

conda install package_name

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
