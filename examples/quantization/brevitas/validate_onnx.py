# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from argparse import ArgumentParser

import random
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoTokenizer

from optimum.amd import BrevitasQuantizationConfig
from optimum.amd.brevitas.data_utils import get_dataset_for_model


@torch.no_grad()
def onnx_compute_perplexity(onnx_file, data, context_length: int, tokenizer, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    ort_sess = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    #ort_sess = ort.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])

    cross_entropy_loss = nn.CrossEntropyLoss()

    nlls = []
    for sample in tqdm(data, desc="Computing perplexity..."):
        sample_length = sample["input_ids"].shape[1]
        for start_index in range(0, sample_length, context_length * 2):
            end_index = min(start_index + sample_length, sample_length - 1)

            subsample = {
                "input_ids": sample["input_ids"][:, start_index : end_index + 1],
                "attention_mask": sample["attention_mask"][:, start_index : end_index + 1],
            }

            # In case we are using torch.fx, we can not have optional inputs, and we have traced the model with past_key_values inputs, thus we need them here as well.
            if "past_key_values" in sample:
                subsample["past_key_values"] = sample["past_key_values"]

            # Add BOS token.
            subsample["input_ids"][:, 0] = tokenizer.bos_token_id

            # Put subsample on CPU
            onnx_subsample = {}
            for k in subsample.keys():
                if isinstance(subsample[k], tuple):
                    for j in range(len(subsample[k])):
                        onnx_subsample[f"past_key_values.{j}.key"] = subsample[k][j][0].cpu().numpy()
                        onnx_subsample[f"past_key_values.{j}.value"] = subsample[k][j][1].cpu().numpy()
                else:
                    onnx_subsample[k] = subsample[k].cpu().numpy()

            #lm_logits = model(**subsample)["logits"]
            onnx_out = ort_sess.run(["logits"], onnx_subsample)
            lm_logits = torch.tensor(onnx_out[0])

            reference_labels = subsample["input_ids"][:, context_length:]

            shift_logits = lm_logits[:, context_length - 1 : -1]

            # Fuse batch and sequence length dimensions.
            reference_labels = reference_labels.view(reference_labels.shape[-1])
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])

            loss = cross_entropy_loss(shift_logits, reference_labels)

            nlls.append(loss)

    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl

def main(args):
    return_val = {}

    onnx_file = f"{args.onnx_path}/model.onnx"
    model_name = args.model
    model_config = AutoConfig.from_pretrained(args.model)
    attention_head_size = int(model_config.hidden_size / model_config.num_attention_heads)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qconfig = BrevitasQuantizationConfig() # Actual contents won't matter
    validation_dataset = get_dataset_for_model(
        args.model,
        qconfig=qconfig,
        dataset_name="wikitext2",
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="validation",
        device="cpu",
        fuse_sequences=args.fuse_sequences,
    )
    
    perplexity = onnx_compute_perplexity(onnx_file, validation_dataset, context_length=args.seqlen // 2, tokenizer=tokenizer)
    print(f"ONNX Perplexity: {perplexity}")
    return_val = {"onnx_ppl"}
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Quantize LLMs from 🤗 Transformers with AMD Brevitas")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model checkpoint to quantize. Can either be a local path to a model folder, or a model on Hugging Face Hub. Example: facebook/opt-125m",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=128,
        help="Sequence length to use during calibration (default: %(default)s).",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of samples to use during calibration & validation (default: %(default)s).",
    )
    parser.add_argument(
        "--fuse-sequences",
        action="store_true",
        default=False,
        help="Whether to merge the dataset sequences in case they are shorter than the requested number of samples per sequence. This is useful in case you would like to quantize or evaluate on long sequences (default: %(default)s).",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="llm_quantized_onnx",
        help="Location of the ONNX model (default: %(default)s)",
    )

    args = parser.parse_args()

    return_val = main(args)
