import os, sys

# example: python train_run.py keyword temp_keyword _
if __name__ == '__main__':
    
    MODEL_FILE = "gpt2-medium"
    
    COMMANDLINE = "python model_diff.py \
        --model_type=gpt2 \
        --length 100 \
        --model_name_or_path={} \
        --sub_model_name_or_path={} \
        --num_return_sequences 5 \
        --stop_token [EOS] \
    ".format(MODEL_FILE, "/home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/trainers/gpt2/webnlg_models/webnlgsubnettune_n_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_sub=both_ratio=1.0_a-spar=1.0_m-spar=1.0_o=1_o=1")

    if MODEL_FILE == 'gpt2-large':
        COMMANDLINE += ' --cache_dir cache/gpt2-large-s3 '

    if MODEL_FILE == 'gpt2-medium':
        COMMANDLINE += ' --cache_dir cache/gpt2-medium-s3 '

    print(COMMANDLINE)

    os.system(COMMANDLINE)