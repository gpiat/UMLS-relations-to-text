#!/bin/bash
# Choisir nb noeuds
#SBATCH -N 1
## nombre de tasks
#SBATCH -n 3
## nombre de threads
##SBATCH -c
#SBATCH --mem=30000
#SBATCH --time=2-00:00:00
# Choisir partition (commande `sinfo` pour voir les differentes partitions)
#SBATCH -p gpuv100,gpup100,gpup6000
# Nb GPU que je veux
#SBATCH --gres=gpu:3
# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=guilhem.piat@cea.fr
# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=bestumls_all
# Limite du nb de CPU

# Chargment de vos modules
#
# Affiche la machine(s)
echo "Begin on machine :"
hostname
# nvidia-smi
source activate snlibaseline

MODEL=$1
NAME=$2

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --ntasks=1 --gres=gpu:1 python baseline_chemprot.py \
../$MODEL/ > 'chemprot_'$NAME'.out' &

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --ntasks=1 --gres=gpu:1 python baseline_cola.py \
../$MODEL/ > 'cola_'$NAME'.out' &

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --ntasks=1 --gres=gpu:1 python baseline_i2b2.py \
../$MODEL/ > 'i2b2_'$NAME'.out' &

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --ntasks=1 --gres=gpu:1 python baseline_snli.py \
../$MODEL/ > 'snli_'$NAME'.out' &

BSIZE=64
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --ntasks=1 --nodes=1 --gres=gpu:1 python baseline_mlm.py \
-m ../$MODEL/ \
-b $BSIZE \
-o '$NAME'-finetuned-mlm > 'mlm_'$NAME'.out' &

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --ntasks=1 --gres=gpu:1 python baseline_bioqa.py \
reasoning-required without-artificial \
../$MODEL/ > 'bioqa_'$NAME'.out' &

wait

# python -u

echo "Done."
# ./ Fin
