#!/bin/bash
#sbatch -W -p ${Partition} -t 2-00:00 --mem 184000 -c 48 -o logs/slurm-%j.out --wrap "source ~/.bashrc; conda activate $CondaEnv; python write_tropomi_GC_daily_avgs.py"; wait;
sbatch -p seas_compute,huce_intel,huce_cascade,test -t 0-00:05 -c 1 -o logs/slurm-%j.out --wrap "python test.py"
