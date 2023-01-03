#!/bin/bash
#SBATCH -e sep_train_bert_models.err
#SBATCH -o sep_train_bert_models.out
# Choisir nb noeuds
#SBATCH -N 1
## nombre tasks
#SBATCH -n 3
## nombre de threads (cpus per task)
#SBATCH -c 8
## le entier rien que pour moi
##SBATCH --exclusive
#SBATCH --mem=24000
#SBATCH --time=4-00:00:00
# Choisir partition (commande `sinfo` pour voir les differentes partitions)
#SBATCH -p gpuv100,gpup100,gpup6000
# Nb GPU que je veux
#SBATCH --gres=gpu:1
# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=guilhem.piat@cea.fr
# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=sep_train_bert_UMLS_gen
# Limite du nb de CPU

# Chargment de vos modules
#
# Affiche la machine(s)
echo "Begin on machine :"
hostname
# nvidia-smi
source activate hfds

TIMESTAMP=`date +"%Y-%m-%d %T" | sed 's/\s/_/g' | sed 's/[:-]//g'`

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --ntasks=1 --gres=gpu:1 python train_models.py \
-d umls \
-o $TIMESTAMP'_UMLS_model/' \
-t 8 &

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --ntasks=1 --gres=gpu:1 python train_models.py \
-d pmc \
-o $TIMESTAMP'_PMC_model/' \
-t 8 &

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
srun --ntasks=1 --gres=gpu:1 python train_models.py \
-d both \
-o $TIMESTAMP'_Hybrid_model/' \
-t 8 &

wait

# python -u

echo "Done."
# ./ Fin
