#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=supervDef
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=logs/%A_%a.stdout
#SBATCH --error=logs/%A_%a.stderr
#SBATCH --array=0-5
#SBATCH --exclude=n[1-5]



SECONDS=0

restart(){
    echo "Calling restart" 
    scontrol requeue $SLURM_JOB_ID
    echo "Scheduled job for restart" 
}

ignore(){
    echo "Ignored SIGTERM" 
}
trap restart USR1
trap ignore TERM

date 



args=()

# regular supervised learning retaining on defender data, w. reasonable HP setting, no overfit
args+=("supervised_train_resnet50_defender.py --train_path data/QMNIST_ppml_ImageFolder/defender --val_path data/QMNIST_ppml_ImageFolder/reserve --batch_size 64 --weight_decay 1e-4 --scheduler_patience 4 --scheduler_factor 0.1 --epochs 40 --random_seed 68 --train_mode fc --num_workers 8")
args+=("supervised_train_resnet50_defender.py --train_path data/QMNIST_ppml_ImageFolder/defender --val_path data/QMNIST_ppml_ImageFolder/reserve --batch_size 64 --weight_decay 1e-4 --scheduler_patience 4 --scheduler_factor 0.1 --epochs 40 --random_seed 68 --train_mode whole --num_workers 8")

# regular supervised learning, train to maximum overfitting, train a long time, no regularization
args+=("supervised_train_resnet50_defender.py --train_path data/QMNIST_ppml_ImageFolder/defender --val_path data/QMNIST_ppml_ImageFolder/reserve --batch_size 64 --weight_decay 0 --scheduler_patience 4 --scheduler_factor 0.1 --epochs 80 --random_seed 68 --train_mode fc --overfit --num_workers 8")
args+=("supervised_train_resnet50_defender.py --train_path data/QMNIST_ppml_ImageFolder/defender --val_path data/QMNIST_ppml_ImageFolder/reserve --batch_size 64 --weight_decay 0 --scheduler_patience 4 --scheduler_factor 0.1 --epochs 80 --random_seed 68 --train_mode whole --overfit --num_workers 8")


# same as the previous one but with labels flipped in both defender and reserve datasets
args+=("supervised_train_resnet50_defender.py --train_path data/QMNIST_ppml_flipped_ImageFolder/defender --val_path data/QMNIST_ppml_flipped_ImageFolder/reserve --batch_size 64 --weight_decay 0 --scheduler_patience 4 --scheduler_factor 0.1 --epochs 80 --random_seed 68 --train_mode fc --overfit --random_labels --num_workers 8")
args+=("supervised_train_resnet50_defender.py --train_path data/QMNIST_ppml_flipped_ImageFolder/defender --val_path data/QMNIST_ppml_flipped_ImageFolder/reserve --batch_size 64 --weight_decay 0 --scheduler_patience 4 --scheduler_factor 0.1 --epochs 80 --random_seed 68 --train_mode whole --overfit --random_labels --num_workers 8")


echo "Starting python ${args[${SLURM_ARRAY_TASK_ID}]}"

srun python ${args[${SLURM_ARRAY_TASK_ID}]}

echo "End python ${args[${SLURM_ARRAY_TASK_ID}]}"



DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 

