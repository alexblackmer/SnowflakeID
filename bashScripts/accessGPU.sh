#!/bin/bash

srun --nodes=1 --ntasks=14 --partition=kingspeak-gpu-guest --account=owner-gpu-guest --gres=gpu:p100:1      --time=24:00:00 --mem=28GB --pty /bin/bash -l

