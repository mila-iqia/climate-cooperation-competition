CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_no_nego.yaml --perturbation -0.01 
CUDA_VISIBLE_DEVICES=1 python /home/ubuntu/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_no_nego.yaml --perturbation -0.02
CUDA_VISIBLE_DEVICES=2 python /home/ubuntu/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_no_nego.yaml --perturbation -0.03
CUDA_VISIBLE_DEVICES=3 python /home/ubuntu/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_no_nego.yaml --perturbation -0.04 
# CUDA_VISIBLE_DEVICES=0 python /home/work/official/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_min_mitigation.yaml --perturbation 0.01 
# CUDA_VISIBLE_DEVICES=1 python /home/work/official/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_min_mitigation.yaml --perturbation 0.02
# CUDA_VISIBLE_DEVICES=2 python /home/work/official/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_min_mitigation.yaml --perturbation 0.03
# CUDA_VISIBLE_DEVICES=3 python /home/work/official/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_min_mitigation.yaml --perturbation 0.04 
# CUDA_VISIBLE_DEVICES=0 python /home/work/official/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_max_mitigation.yaml --perturbation 0.01 
# CUDA_VISIBLE_DEVICES=1 python /home/work/official/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_max_mitigation.yaml --perturbation 0.02
# CUDA_VISIBLE_DEVICES=2 python /home/work/official/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_max_mitigation.yaml --perturbation 0.03
# CUDA_VISIBLE_DEVICES=3 python /home/work/official/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_max_mitigation.yaml --perturbation 0.04 
# CUDA_VISIBLE_DEVICES=0 python ~/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_basic_club.yaml --perturbation -0.01 
# CUDA_VISIBLE_DEVICES=0 python ~/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_basic_club.yaml --perturbation -0.02
# CUDA_VISIBLE_DEVICES=0 python ~/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_basic_club.yaml --perturbation -0.03
# CUDA_VISIBLE_DEVICES=0 python ~/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_basic_club.yaml --perturbation -0.04 
# sh ~/ai4gcc-virginia/climate-cooperation-competition/terminate_lambda.sh
# CUDA_VISIBLE_DEVICES=0 python ~/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_bilateral.yaml --perturbation 0.01 
# CUDA_VISIBLE_DEVICES=0 python ~/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_bilateral.yaml --perturbation 0.02
# CUDA_VISIBLE_DEVICES=0 python ~/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_bilateral.yaml --perturbation 0.03
# CUDA_VISIBLE_DEVICES=0 python ~/ai4gcc-virginia/climate-cooperation-competition/scripts/train_with_rllib.py --yaml rice_rllib_discrete_bilateral.yaml --perturbation 0.04 
# sh ~/ai4gcc-virginia/climate-cooperation-competition/terminate_lambda.sh