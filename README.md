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

# Hyperparameters for Diffusion Model Training

You can view and modify these hyperparameters in the program "train_generate_E_coli_promoter/train_generate_model.py". Below are the descriptions of the hyperparameters used in the diffusion model training script:

### 1. `learning_rate`
- **Description**: Controls the rate at which the model responds to the loss function in each iteration.
- **Default Value**: `1e-3` (0.001)

### 2. `batch_size`
- **Description**: The number of samples used in each training iteration.
- **Default Value**: `256`
- **Note**: A larger `batch_size` can improve training stability but requires more memory. Users adjust the batch size according to their GPU’s memory constraints

### 3. `iterations`
- **Description**: Total number of training iterations.
- **Default Value**: `2000`

### 4. `log_to_wandb`
- **Description**: Enables logging to Weights & Biases (WandB) for experiment tracking.
- **Default Value**: `True` (enabled)

### 5. `log_rate`
- **Description**: Frequency of logging. This defines how often logs are recorded during training.
- **Default Value**: `1` (logs after each iteration)

### 6. `checkpoint_rate`
- **Description**: Frequency of model checkpoint saving.
- **Default Value**: `100` (saves checkpoint every 100 iterations)

### 7. `log_dir`
- **Description**: Directory for saving logs and model checkpoints.
- **Default Value**: `"../model"`

### 8. `project_name`
- **Description**: Name of the project, used for logging and saving model files.
- **Default Value**: `'cyanobacteria'`

### 9. `out_init_conv_padding`
- **Description**: Initial padding size for convolutional layers in the model. Controls the effective area of convolution kernels.
- **Default Value**: `1`

### 10. `run_name`
- **Description**: Name for the current run, typically with a timestamp for unique identification of different runs.
- **Default Value**: Automatically generated, e.g., `"ddpm-2024-11-10-15-30"`

### 11. `model_checkpoint`
- **Description**: Path to a model checkpoint file for resuming training or inference. 
- **Default Value**: `None` (starts training from scratch)

### 12. `optim_checkpoint`
- **Description**: Path to an optimizer checkpoint file for resuming training.
- **Default Value**: `None` (starts with a new optimizer)

### 13. `schedule_low`
- **Description**: Lower limit for the learning rate scheduler, defining the minimum learning rate.
- **Default Value**: `1e-4` (0.0001)

### 14. `schedule_high`
- **Description**: Upper limit for the learning rate scheduler, defining the maximum learning rate.
- **Default Value**: `0.02`

The approximate training time for the diffusion model with default parameters on an 80GB A800 GPU is around 24 hours.
