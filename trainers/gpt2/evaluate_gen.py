import os, sys

# example: python train_run.py keyword temp_keyword _
if __name__ == '__main__':
    mode = sys.argv[1]
    control_mode = sys.argv[2]
    eval_split = sys.argv[3]
    model_file = sys.argv[4]
    old_model = None
    MODEL_FILE = "gpt2-medium"
    submit_job = (sys.argv[5] == 'yes')
    a_spar = sys.argv[6]
    m_spar = sys.argv[7]


    if mode =='data2text':

        Token_FILE = MODEL_FILE

        # gen_dir = 'e2e_results_conv'
        gen_dir = 'e2e_results_conv2'

        sub_model_name = os.path.basename(MODEL_FILE)
        if 'checkpoint-' in sub_model_name:
            sub_model_name =  MODEL_FILE
        
        tuning_mode = 'finetune'
        app = ''

    elif mode == 'writingPrompts' or mode == 'sentiment' or mode == 'topic':
        Token_FILE = MODEL_FILE
        if mode == 'writingPrompts':
            gen_dir = 'wp_results'
        else:
            gen_dir = 'class_conditional_results'

        tuning_mode = 'finetune'
        app = ''


    elif mode == 'classify-sentiment' or mode == 'classify-topic':

        Token_FILE = MODEL_FILE
        sub_model_name = os.path.basename(MODEL_FILE)

        gen_dir = 'classify_results'

        tuning_mode = 'finetune'
        app = ''

    elif mode == 'triples':
        Token_FILE = MODEL_FILE

        gen_dir = 'triples_results'
        sub_model_name = os.path.basename(MODEL_FILE)

        tuning_mode = 'finetune'
        app = ''

    elif mode == 'webnlg':
        Token_FILE = MODEL_FILE
        # gen_dir = 'webNLG_results'
        # gen_dir = 'webNLG_results2'
        gen_dir = "webnlg_evals"

        tuning_mode = 'finetune'
        app = ''

    elif mode == 'cnndm' or mode == 'xsum':
        Token_FILE = MODEL_FILE
        gen_dir = 'xsum_results2'

        sub_model_name = os.path.basename(MODEL_FILE)

        tuning_mode = 'finetune'
        app = ''

    COMMANDLINE = "python run_eval_gen.py \
        --model_type=gpt2 \
        --length 100 \
        --model_name_or_path={} \
        --model_file={} \
        --num_return_sequences 5 \
        --stop_token [EOS] \
        --tokenizer_name={} \
        --task_mode={} \
        --control_mode={} --tuning_mode {} --gen_dir {} --eval_dataset {} \
        --sparsity_attn={} --sparsity_mlp={} \
    ".format(MODEL_FILE, model_file, Token_FILE, mode, control_mode, tuning_mode, gen_dir, eval_split, a_spar, m_spar)

    COMMANDLINE += app

    if MODEL_FILE == 'gpt2-large':
        COMMANDLINE += ' --cache_dir cache/gpt2-large-s3 '

    if MODEL_FILE == 'gpt2-medium':
        COMMANDLINE += ' --cache_dir cache/gpt2-medium-s3 '


    print(COMMANDLINE)

    os.system(COMMANDLINE)

    # if not submit_job:
    #     os.system(COMMANDLINE)
    # else:
    #     # name = 'e2e_results_lowdata/{}'.format(name)
    #     # name = 'e2e_results_lowdata_finetune/{}'.format(name)
    #     name = os.path.join(gen_dir, name)
    #     full_command = "nlprun -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard28,jagupard29 \'{}\'".format(name,COMMANDLINE)
    #     print(full_command)
    #     os.system(full_command)