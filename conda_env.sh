#!/bin/bash

# Initialize Conda for this script
eval "$(conda shell.bash hook)"

# Define environment name and Python version
ENV_NAME=top_erl_iclr25
PYTHON_VERSION=3.11

# Create a new conda environment
echo "Creating a new conda environment named $ENV_NAME with Python $PYTHON_VERSION"
conda create --name $ENV_NAME python=$PYTHON_VERSION -y

# Activate the newly created environment
echo "Activating the $ENV_NAME environment"
conda activate $ENV_NAME

# Verify if the correct conda environment is activated
echo
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "$CONDA_DEFAULT_ENV"
    echo Failed to activate conda environment.
    exit 1
else
    echo Successfully activated conda environment.
fi

# Install mamba release to boost installation and resolve dependencies
# conda install -c conda-forge mamba=1.4.2 -y


# Install packages using conda or mamba
echo "Installing packages with conda or mamba"
conda install -c hussamalafandi cppprojection -c conda-forge -y
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install wandb=0.16.3 -y
mamba install natsort=8.4.0 -y
mamba install tabulate=0.9.0 -y
mamba install conda-build=24.1.2 -y
mamba install matplotlib -y
mamba install tqdm -y
mamba install addict -y
mamba install numpy=1.26 -y
pip install toml==0.10.2

# Add the current repo to conda env python path
conda develop .

cd ..
# Download packages from Github

# Fancy_Gym
git clone -b iclr25_final git@github.com:BruceGeLi/fancy_gymnasium.git
cd fancy_gymnasium
pip install -e .
conda develop .
cd ..

# Trust_Region_Projection
git clone -b iclr25_final git@github.com:BruceGeLi/trust-region-layers.git
cd trust-region-layers
conda develop .
cd ..

# Git_Repo_Tracker
git clone -b iclr25_final git@github.com:ALRhub/Git_Repos_Tracker.git
cd Git_Repos_Tracker
pip install -e .
conda develop .
cd ..

# MetaWorld
git clone -b iclr25_final git@github.com:BruceGeLi/Metaworld.git
cd Metaworld
pip install -e .
conda develop .
cd ..

# cw2
git clone -b iclr25_final git@github.com:BruceGeLi/cw2.git
cd cw2
pip install -e .
conda develop .
cd ..

# MP_PyTorch
git clone -b iclr25_final git@github.com:ALRhub/MP_PyTorch.git
cd MP_PyTorch
conda develop .
cd ..

pip install stable-baselines3==2.2.1

echo "Configuration completed successfully."