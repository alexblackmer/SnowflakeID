#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --partition=kingspeak-gpu-guest
#SBATCH --account=owner-gpu-guest
#SBATCH --gres=gpu:p100:2
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --pty /bin/bash -l

salloc -n 16 -N 1 -t 2:00:00 -p kingspeak-gpu-guest -A owner-gpu-guest --gres=gpu:p100:2 --mem=0


