#!/bin/bash

# 假设这是你的bash文件的路径
script_path="/home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/dart/evaluation/run_eval_on_webnlg.sh"

echo "Start..."

# 循环遍历每个sparsity值
for sub in "attn"; do
    for spar in 0.1; do
        output_file="/home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/trainers/gpt2/webnlg_models/webnlgsubnettune_n_10_act_cat_b=15-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_sub=${sub}_ratio=1.0_spar=${spar}_o=1_o=1_test_beam"
        teamr="webnlgsubnettune_n_10_act_cat_b=15-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_sub=${sub}_ratio=1.0_spar=${spar}_o=1_o=1_test_beam"

        output_result="/home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/trainers/gpt2/webnlg_models/webnlgsubnettune_n_10_act_cat_b=15-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_sub=${sub}_ratio=1.0_spar=${spar}_o=1_o=1_test_beam_eval"
        # 调用脚本并传递参数
        bash $script_path $output_file $teamr >> $output_result

        echo "Finish with $sub and $spar."
    done
done