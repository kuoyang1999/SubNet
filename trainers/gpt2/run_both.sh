#!/bin/bash

# 初始化sparsity的值
sparsity=0.1

# 持续监控GPU 1的状态
while true; do

    # 如果sparsity达到1.0，终止脚本
    if (( $(echo "$sparsity >= 1.0" | bc -l) )); then
        echo "sparsity reached 1.0, stopping execution."
        break
    fi

    # 获取GPU 1的利用率
    usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk 'NR==2')

    # 检查GPU 1的利用率是否低于10%
    if [[ $usage -lt 10 ]]; then
        echo "GPU 1 is idle, running command with sparsity $sparsity..."
        # 运行命令，每次使用新的sparsity值
        CUDA_VISIBLE_DEVICES=1 COMET_MODE=DISABLED python train_e2e.py --tuning_mode subnettune --optim_prefix no --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 10 --seed 101 --subnet_mode both --sparsity $sparsity --sub_ratio 1.0
        # 更新sparsity值
        sparsity=$(echo "$sparsity + 0.1" | bc)
    else
        echo "GPU 1 is busy."
    fi

    echo "Checking again in 60 seconds..."
    sleep 60  # 每60秒检查一次

done