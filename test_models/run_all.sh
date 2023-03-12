#!/bin/bash

#for MODEL in 20230308_231002_PMC123_model 20230308_232115_HybridFull_model 20230308_232115_HybridHalf_model; do
for MODEL in ../20230308_231002_PMC123_model ../20230308_232115_HybridFull_model ../20230308_232115_HybridHalf_model ../20230304_010627_UMLS_model_lr2e-6_mlm0.15_1ep bert-base-uncased; do
    NAME=`echo $MODEL | cut -d '_' -f 3`
    sbatch --error='all_'$NAME'.err' --output='all_'$NAME'.out' run_all.slurm $MODEL $NAME
done
