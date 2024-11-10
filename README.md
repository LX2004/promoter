# Platform
To directly use the trained model, please visit our [Promoter Design Integration Platform](https://bioinformatics-syn.org/). After a brief setup and a short wait, you will receive the desired results.

# Pre-trainde model
[Escherichia coli Promoter Generation Model](https://drive.google.com/file/d/1S7PESHCbILT_Z_rodAT_2TA1d93se2LV/view?usp=drive_link); 

[Cyanobacteria Promoter Generation Model](https://drive.google.com/file/d/1SvVBMARE96mMdp8DzGCkfkzZTouJIdFt/view?usp=drive_link); 

[Predictive Model](https://drive.google.com/file/d/1De7xCmCfwCoYH_zECsprUTsSUeROtg00/view?usp=drive_link).

Click the link to download the pre-trained model.

Models used by other researchers for promoter generation can be accessed through the following link: [VAE](https://figshare.com/articles/software/CyanoDeeplearning/2233%E2%80%8C1044);[GAN](https://github.com/HaochenW/Deep_promoter).

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

python generate_promoters.py
```
# For prediction task
run : 
```
cd train_prediction_model

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

### 15. `early_stopping`
- **Description**: Early stopping parameter.
- **Default Value**: `10`
- 
The approximate training time for the diffusion model with default parameters on an 80GB A800 GPU is around 12 hours.

# Hyperparameter Explanation for predictive model

This document provides explanations for the hyperparameters used in the predictive model. Each hyperparameter has a specific role in the training process and model configuration. Below is a detailed description of each one:

---

### Device Configuration

- **`device_num`**: `0`
  - Specifies the GPU device number to be used for training. In this case, it refers to the GPU with ID 4. Ensure that the system has the appropriate GPU and that it is available for training.

---

### Dropout Parameters

- **`dropout_rate1`**: `0.3254948178441311`
  - The dropout rate for the transformer1 which is used to process original sequence information. Dropout is used to prevent overfitting by randomly setting a fraction of input units to zero during training.

- **`dropout_rate2`**: `0.36751719371886576`
  - The dropout rate for the transformer2 which is used to process dinucleotides information.

- **`dropout_rate_fc`**: `0.4458100938040957`
  - The dropout rate for the fully connected (fc) layers.

---

### Embedding Dimensions

- **`embedding_dim1`**: `64`
  - The dimension of the transformer1. This is the size of the vector representation for each input token in the model.

- **`embedding_dim2`**: `64`
  - The dimension of the transformer2. This helps in transforming the data into a suitable representation for the following layers of the model.

---

### Fully Connected Layer Parameters

- **`fc_hidden1`**: `210`
  - The number of units (neurons) in the first fully connected (dense) layer. It represents the complexity of the model in terms of the number of hidden neurons.

- **`fc_hidden2`**: `37`
  - The number of units in the second fully connected layer. It follows the first fully connected layer and helps in feature transformation.

---

### Hidden Layer Parameters

- **`hidden_dim1`**: `128`
  - The number of neurons in the hidden layer of the transformer1. This is an important parameter that controls the capacity of the model at this stage.

- **`hidden_dim2`**: `1024`
  - The number of neurons in the hidden layer of the transformer2. This layer is larger and is designed to allow for more complex transformations.

---

### Regularization

- **`l2_regularization`**: `1e-5`
  - This is the L2 regularization parameter (also known as weight decay). It helps prevent overfitting by penalizing large weights in the model. A smaller value indicates less regularization, while larger values impose stronger penalties.

---

### Latent Dimensions

- **`latent_dim1`**: `64`
  - The dimensionality of the transformer1. This represents the size of the compressed representation in the latent space.

- **`latent_dim2`**: `256`
  - The dimensionality of the transformer2. A larger latent dimension can capture more complex representations but may also lead to overfitting.

---

### Attention Mechanism (Transformer)

- **`num_head1`**: `8`
  - The number of attention heads in the first attention layer of the transformer1. More heads allow the model to focus on different parts of the input sequence, capturing more information.

- **`num_head2`**: `16`
  - The number of attention heads in the first attention layer of the transformer2. This helps the model learn better representations from the input data.

---

### Sequence Length

- **`seq_len`**: `50`
  - The length of the input promoter sequences.

---

### Training Parameters

- **`train_base_learning_rate`**: `0.0001`
  - The base learning rate used in training. This controls how quickly the model updates its weights during training. A smaller value ensures more stable updates.

- **`train_batch_size`**: `512`
  - The batch size used in training. A larger batch size can improve training efficiency but requires more memory.

- **`train_epochs_num`**: `500`
  - The number of epochs (iterations over the entire dataset) for which the model will be trained. Increasing the number of epochs allows the model more opportunities to learn, but it also increases computational cost.

---

### Transformer Model Parameters

- **`transformer_num_layers1`**: `3`
  - The number of layers of the transformer1 model. Each layer contains attention and feed-forward components that help capture different levels of information.

- **`transformer_num_layers2`**: `3`
  - The number of layers of the transformer1 mode2. Increasing the number of layers can increase model capacity but also computational complexity.

---

This file aims to provide a clear understanding of the hyperparameters used for the model. Adjusting these parameters can significantly impact the performance and efficiency of the model.

# References
1. Synthetic promoter design in Escherichia coli based on a deep generative network.

2. Design of synthetic promoters for cyanobacteria with generative deep-learning model.
