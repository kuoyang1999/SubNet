        # # should clone the config and construct it.
        # config_subnet = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        # config_subnet._my_arg_tune_mode = model_args.tuning_mode
        # config_subnet._my_arg_task_mode = data_args.task_mode
        # config_subnet._my_arg_control = True
        # config_subnet.train_weights = data_args.train_embs
        # config_subnet.optim_prefix = optim_prefix_bool
        # config_subnet.preseqlen = model_args.preseqlen
        # config_subnet.use_infix = (data_args.format_mode == 'infix')
        # config_subnet.format_mode = data_args.format_mode
        # config_subnet.prefix_dropout = model_args.prefix_dropout
        # config_subnet.vocab_size = len(tokenizer)
        # config_subnet.lowdata = ('lowdata' in training_args.output_dir)
        # if not config_subnet.lowdata:
        #     config_subnet.lowdata = ('datalevels' in training_args.output_dir and data_args.use_lowdata_token == 'yes')
        # if config_subnet.lowdata and data_args.use_lowdata_token == 'yes':
        #     config_subnet.lowdata_token = tokenizer([data_args.lowdata_token],
        #                                             add_prefix_space=True)['input_ids']  #return_tensors='np',
        #     print(data_args.lowdata_token)
        #     print(config_subnet.lowdata_token)

        # # some extra stuff.
        # config_subnet.init_random = model_args.init_random
        # config_subnet.mid_dim = model_args.mid_dim
        # config_subnet.init_shallow = model_args.init_shallow
        # if config_subnet.init_shallow == 'yes':
        #     if model_args.init_shallow_word != 'no':
        #         config_subnet.init_shallow_word = tokenizer([model_args.init_shallow_word],
        #                                                     add_prefix_space=True)['input_ids']  #return_tensors='np',
        #     else:
        #         config_subnet.init_shallow_word = None
        #     print(model_args.init_shallow_word)
        #     print(config_subnet.init_shallow_word)

CUDA_VISIBLE_DEVICES=0 COMET_MODE=DISABLED python Evaluation_SubNet.py --output_dir webnlg_evals/webnlgeval_sub=both_a-spar=1.0_m-spar=1.0 --model_type gpt2 --model_name_or_path /home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/trainers/gpt2/webnlg_models/webnlgsubnettune_n_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_sub=both_ratio=1.0_a-spar=1.0_m-spar=1.0_o=1_o=1 --tokenizer_name /home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/trainers/gpt2/webnlg_models/webnlgsubnettune_n_10_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_sub=both_ratio=1.0_a-spar=1.0_m-spar=1.0_o=1_o=1 --per_device_eval_batch_size 32 --do_eval --line_by_line --overwrite_output_dir --task_mode webnlg --eval_data_file /home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/data/webnlg_challenge_2017/dev.json --tuning_mode finetune --logging_dir webnlg_evals/runs/webnlgeval_sub=both_a-spar=1.0_m-spar=1.0 --sparsity_attn 1.0 --sparsity_mlp 1.0 --subnet_mode both --sub_ratio 1.0 --sparsity 1.0 --tuning_mode_sub subnettune --train_data_file /home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/data/webnlg_challenge_2017/train.json