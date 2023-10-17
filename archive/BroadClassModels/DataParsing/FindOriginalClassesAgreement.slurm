#!/bin/csh

#SBATCH --ntasks=2 
#SBATCH --mem=8GB 
#SBATCH --time=8:00:00
#SBATCH --account=garrett
#SBATCH --partition=kingspeak-shared


setenv WORKDIR $HOME/Research/SnowflakeID/InputData
source ../venv/bin/activate.csh
python FindOriginalClassesAgreement.py
