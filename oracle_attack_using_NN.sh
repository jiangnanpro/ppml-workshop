#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=WBattack
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --cpus-per-task=3
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=logs/%A_%a.stdout
#SBATCH --error=logs/%A_%a.stderr
#SBATCH --array=0
#SBATCH --exclude=n[1-5,51-55]



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



# (white-box attacker using a neural network) forward_target_domain, regular supervised learning retaining on defender data, w. reasonable HP setting, no overfit
## whole
args+=("oracle_attack_using_NN.py --model_path supervised_model_checkpoints/best_model_supervised_resnet50_QMNIST_defender_whole-0.0001-normal-normal_gallant-wildflower-1.pth --results_dir supervised_normal_whole_attack_using_NN --dataset_path data/QMNIST_ppml.pickle --attack_mode forward_target_domain --N 3000 --zip")



echo "Starting python ${args[${SLURM_ARRAY_TASK_ID}]}"

srun python ${args[${SLURM_ARRAY_TASK_ID}]}

echo "End python ${args[${SLURM_ARRAY_TASK_ID}]}"



DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 

