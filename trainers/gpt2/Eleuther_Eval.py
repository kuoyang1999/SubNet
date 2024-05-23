#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    AutoConfig,
    set_seed,
)
from gptSubNet import GPT2LMHeadModel_SubNet

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}
MODEL_CLASSES_SUBNET = {
    "gpt2": (GPT2LMHeadModel_SubNet, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

set_seed(101)

# Initialize the model and tokenizer
model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
print('loading the gpt2 tokenizer')
tokenizer = tokenizer_class.from_pretrained("gpt2-medium", cache_dir="/home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/trainers/gpt2/cache/gpt2-medium-s3")
print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
config = AutoConfig.from_pretrained("gpt2-medium", cache_dir="/home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/trainers/gpt2/cache/gpt2-medium-s3")
config._my_arg_tune_mode = "subnettune"
config._my_arg_task_mode = "webnlg"
config._objective_mode = 2
print(config)
model_ori = model_class.from_pretrained("gpt2-medium", config=config, cache_dir="/home/mingyuan/Lab/Mingyuan/SubNet/PrefixTuning/trainers/gpt2/cache/gpt2-medium-s3")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model_ori.resize_token_embeddings(len(tokenizer))
print(model_ori)

model_class, _ = MODEL_CLASSES_SUBNET["gpt2"]
config = AutoConfig.from_pretrained("webnlg_models/webnlgsubnettune_n_10_act_cat_b=10-e=3_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_sub=both_ratio=1.0_a-spar=0.8_m-spar=0.5_o=1_o=1", cache_dir=None)
print(config)
model_trained = model_class.from_pretrained("webnlg_models/webnlgsubnettune_n_10_act_cat_b=10-e=3_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_sub=both_ratio=1.0_a-spar=0.8_m-spar=0.5_o=1_o=1", config=config, cache_dir=None)
print(model_trained)

ori_param = dict(model_ori.named_parameters())
trained_param = dict(model_trained.named_parameters())
for name, param in trained_param.items():
    if "attn" in name or "mlp" in name:
        if name in ori_param.keys():
            assert torch.equal(param.detach(), ori_param[name].detach()), f"{name} in the trained model is not equal to the ori model."
            
            mask_name = name.replace('weight', 'scores')
            score = trained_param.get(mask_name, None)
            assert score is not None, f"There's no mask for {name}."
            
            score = score.abs()
            mask = score.clone()
            _, idx = score.flatten().sort()
            if "attn" in name:
                j = int((1 - 0.8) * score.numel())
                flat_mask = mask.flatten()
                flat_mask[idx[:j]] = 0
                flat_mask[idx[j:]] = 1
            elif "mlp" in name:
                j = int((1 - 0.5) * score.numel())
                flat_mask = mask.flatten()
                flat_mask[idx[:j]] = 0
                flat_mask[idx[j:]] = 1

            with torch.no_grad():
                ori_param[name].copy_(ori_param[name] * mask)
        else:
            assert name.split(".")[-1] == "scores"
    else:
        assert "scores" not in name, f"Error, {name} has a mask."
        # with torch.no_grad():
        #     ori_param[name].copy_(param)

# Re-assign modified parameters back to the original model
with torch.no_grad():
    for name, param in model_ori.named_parameters():
        if name in ori_param:
            param.copy_(ori_param[name])

model_ori.save_pretrained('./saved-for-eval/a08-m05-e3')
tokenizer.save_pretrained("./saved-for-eval/a08-m05-e3/tokenizer/")