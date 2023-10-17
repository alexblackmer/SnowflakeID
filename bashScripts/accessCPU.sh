#!/bin/bash

srun --nodes=1 --ntasks=2 --partition=kingspeak-shared --account=garrett --mem=16GB --pty /bin/bash -l

