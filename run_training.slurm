#!/bin/bash
#SBATCH -e train_bert_models.err
#SBATCH -o train_bert_models.out
# Choisir nb noeuds
#SBATCH -N 1
## nombre processus
#SBATCH -n 1
## nombre de threads
#SBATCH -c 8
## le entier rien que pour moi
##SBATCH --exclusive
#SBATCH --mem=24000
#SBATCH --time=0-04:00:00
# Choisir partition (commande `sinfo` pour voir les differentes partitions)
#SBATCH -p gpuv100,gpup100,gpup6000
# Nb GPU que je veux
#SBATCH --gres=gpu:1
# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=guilhem.piat@cea.fr
# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=train_bert_UMLS_gen
# Limite du nb de CPU

# Chargment de vos modules
#
# Affiche la machine(s)
echo "Begin on machine :"
hostname
# nvidia-smi
source activate hfds

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --gres=gpu:1 python train_models.py

wait

# python -u

echo "Done."
# ./ Fin
