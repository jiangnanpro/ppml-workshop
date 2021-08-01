#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=suDef
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --cpus-per-task=3
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=logs/%A_%a.stdout
#SBATCH --error=logs/%A_%a.stderr
#SBATCH --array=0-11



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

# forward_target_domain, DA
## small Fake-MNIST
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/resnet50_fm_defender.pth --results_dir small_fake_mnist__attack_mode_forward_target_domain --dataset_path data/QMNIST_ppml.pickle --attack_mode forward_target_domain --N 3000 --zip")
## large Fake-MNIST
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/resnet50_large_fm_defender.pth --results_dir large_fake_mnist__attack_mode_forward_target_domain --dataset_path data/QMNIST_ppml.pickle --attack_mode forward_target_domain --N 3000 --zip")


# forward_target_domain, regular supervised learning retaining on defender data, w. reasonable HP setting, no overfit
## fc
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/best_model_supervised_resnet50_QMNIST_defender_fc-0.0001-normal-normal_brisk-sky-4.pth --results_dir supervised_normal_fc --dataset_path data/QMNIST_ppml.pickle --attack_mode forward_target_domain --N 3000 --zip")
## whole
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/best_model_supervised_resnet50_QMNIST_defender_whole-0.0001-normal-normal_gallant-wildflower-1.pth --results_dir supervised_normal_whole --dataset_path data/QMNIST_ppml.pickle --attack_mode forward_target_domain --N 3000 --zip")

# forward_target_domain, regular supervised learning, train to maximum overfitting, train a long time, no regularization
## fc
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/last_model_supervised_resnet50_QMNIST_defender_fc-0.0-normal-overfit_absurd-shadow-2.pth --results_dir supervised_long_fc --dataset_path data/QMNIST_ppml.pickle --attack_mode forward_target_domain --N 3000 --zip")
## whole
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/last_model_supervised_resnet50_QMNIST_defender_whole-0.0-normal-overfit_giddy-yogurt-2.pth --results_dir supervised_long_whole --dataset_path data/QMNIST_ppml.pickle --attack_mode forward_target_domain --N 3000 --zip")

# forward_target_domain, same as the previous one but with labels flipped in both datasets
## fc
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/last_model_supervised_resnet50_QMNIST_defender_fc-0.0-flipped-overfit_jumping-eon-6.pth --results_dir supervised_flipped_fc --dataset_path data/flipped --attack_mode forward_target_domain --N 3000 --zip")
## whole
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/last_model_supervised_resnet50_QMNIST_defender_whole-0.0-flipped-overfit_easy-sponge-5.pth --results_dir supervised_flipped_whole --dataset_path data/flipped --attack_mode forward_target_domain --N 3000 --zip")


# transfer_loss, DA
## small Fake-MNIST
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/resnet50_fm_defender.pth --results_dir small_fake_mnist__attack_mode_transfer_loss --dataset_path data/QMNIST_ppml.pickle --attack_mode transfer_loss --N 3000 --zip --num_workers 3 --source_path data/MNIST_like_")
## large Fake-MNIST
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/resnet50_large_fm_defender.pth --results_dir large_fake_mnist__attack_mode_transfer_loss --dataset_path data/QMNIST_ppml.pickle --attack_mode transfer_loss --N 3000 --zip --num_workers 3 --source_path data/Fake-MNIST-large-28x28x1")


# total_loss, DA
## small Fake-MNIST
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/resnet50_fm_defender.pth --results_dir small_fake_mnist__attack_mode_total_loss --dataset_path data/QMNIST_ppml.pickle --attack_mode total_loss --N 3000 --zip --num_workers 3 --source_path data/MNIST_like_")
## large Fake-MNIST
args+=("oracle_attack_UDA.py --model_path supervised_model_checkpoints/resnet50_large_fm_defender.pth --results_dir large_fake_mnist__attack_mode_total_loss --dataset_path data/QMNIST_ppml.pickle --attack_mode total_loss --N 3000 --zip --num_workers 3 --source_path data/Fake-MNIST-large-28x28x1")





echo "Starting python ${args[${SLURM_ARRAY_TASK_ID}]}"

srun python ${args[${SLURM_ARRAY_TASK_ID}]}

echo "End python ${args[${SLURM_ARRAY_TASK_ID}]}"



DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 

