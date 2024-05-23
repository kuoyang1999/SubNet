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


import argparse
import logging

import numpy as np
import torch
import json

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
    BertForMaskedLM, BertModel,
    BertTokenizer, BertTokenizerFast, AutoConfig,
    set_seed,
    GPT2LMHeadModelAdapter,
)
from gptSubNet import GPT2LMHeadModel_SubNet
import sys, os
from train_control import PrefixTuning, PrefixEmbTuning


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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

#
# Functions to prepare models' input
#

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--sub_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained tokenizer or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--task_mode", type=str, default="embMatch")
    parser.add_argument("--control_mode", type=str, default="yes")
    parser.add_argument("--prefix_mode", type=str, default="activation")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--gen_dir", type=str, default="e2e_results_conv")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--tuning_mode", type=str, default="finetune", help="prefixtune or finetune")
    parser.add_argument("--eval_dataset", type=str, default="val", help="val or test")
    parser.add_argument("--objective_mode", type=int, default=2)
    parser.add_argument("--format_mode", type=str, default="peek", help="peek, cat, nopeek, or infix")
    parser.add_argument("--optim_prefix", type=str, default="no", help="optim_prefix")
    parser.add_argument("--preseqlen", type=int, default=5, help="preseqlen")

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--control_dataless", type=str, default="no", help="control dataless mode")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args.seed)

    # Initialize the model and tokenizer

    print(args.tuning_mode, args.model_name_or_path)
    print(args.cache_dir)
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")
    
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    config._my_arg_tune_mode = 'finetune'
    config._my_arg_task_mode = "webnlg"
    print(config)
    model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)



    # print(args.tuning_mode, args.model_name_or_path)
    # try:
    #     args.model_type = args.model_type.lower()
    #     model_class, _ = MODEL_CLASSES_SUBNET[args.model_type]
    # except KeyError:
    #     raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    # config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=None)
    # print(config)
    # model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=None)



    print(args.tuning_mode, args.sub_model_name_or_path)
    try:
        args.model_type = args.model_type.lower()
        model_class, _ = MODEL_CLASSES_SUBNET[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    config = AutoConfig.from_pretrained(args.sub_model_name_or_path, cache_dir=None)
    print(config)
    model_sub = model_class.from_pretrained(args.sub_model_name_or_path, config=config, cache_dir=None)



    model1_layers = {name: para for name, para in model.named_parameters()}
    model2_layers = {name: para for name, para in model_sub.named_parameters()}

    # print("model1_layers:", model1_layers)
    # print("model2_layers:", model2_layers)

    layer_in1 = 0
    layer_in2 = 0
    for name in model2_layers.keys():
        if name in model1_layers.keys():
            if "mlp" in name or "attn" in name:
                if "weight" in name:
                    layer_in1 += 1
            layer1 = model1_layers[name]
            layer2 = model2_layers[name]
            if layer1.shape != layer2.shape:
                print(name)
                weight1_np = layer1.detach().numpy().flatten()
                weight2_np = layer1.detach().numpy().flatten()
                set_diff1 = np.setdiff1d(weight1_np, weight2_np)
                set_diff2 = np.setdiff1d(weight2_np, weight1_np)
                if len(set_diff1) == 1 and len(set_diff2) == 1:
                    print("只有一个元素不同，分别是：")
                    print("在 weight1 中但不在 weight2 中的元素：", set_diff1)
                    print("在 weight2 中但不在 weight1 中的元素：", set_diff2)
                else:
                    print("两个权重差异较大，不止一个元素不同。")
            else:
                weight_diff = (layer1 - layer2).abs().sum().item()
                print(f"Difference in weights for {name}: {weight_diff}")
        else:
            if "attn" in name or "mlp" in name:
                if "score" in name:
                    # print(model2_layers[name].flatten())
                    layer_in2 += 1
            print(f"Layer {name} not found in model1")

    print("The mask has {} layers.".format(layer_in2/layer_in1))

if __name__ == "__main__":
    main()