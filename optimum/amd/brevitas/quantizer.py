# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import inspect
import logging
from typing import Dict, List, Optional, Union

import torch
from brevitas_examples.common.generative.quantize import quantize_model
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration

# TODO: rather use Optimum/GPTQ data handlers.
from brevitas_examples.llm.llm_quant.data import get_c4, get_wikitext2
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization, apply_weight_equalization
from brevitas_examples.llm.llm_quant.gptq import apply_gptq
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_mha_with_quantizable_layers

from optimum.exporters import TasksManager
from optimum.quantization_base import OptimumQuantizer
from transformers.utils.fx import symbolic_trace

from .configuration import BrevitasQuantizationConfig


logger = logging.getLogger(__name__)


class BrevitasQuantizer(OptimumQuantizer):
    """
    Handles the Runtime quantization process for models shared on huggingface.co/models.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.config = self.model.config

    def validate_quant_config(self, quantization_config: BrevitasQuantizationConfig):
        dtype = next(iter(self.model.parameters()))
        if dtype == torch.bfloat16 and quantization_config.replace_mha_with_quantizable:
            raise RuntimeError("Scaled_dot_product does not support bfloat16 and cuda")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ):
        """
        TODO: doc
        """
        task = TasksManager.infer_task_from_model(model_name_or_path)

        model = TasksManager.get_model_from_task(
            task,
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        return cls(model)

    def quantize(
        self, quantization_config: BrevitasQuantizationConfig, calibration_dataloader: Optional[List[Dict]] = None
    ) -> torch.nn.Module:
        """
        TODO: doc
        """
        self.validate_quant_config(quantization_config)

        requires_data = (
            quantization_config.activations_equalization
            or quantization_config.apply_gptq
            or quantization_config.is_static
        )
        if calibration_dataloader is None and requires_data:
            raise ValueError(
                f"No calibration_dataloader was passed, but a calibration dataset is required with the quantization configuration activations_equalization={quantization_config.activations_equalization}, apply_gptq={quantization_config.apply_gptq}, is_static={quantization_config.is_static}."
            )

        if quantization_config.activations_equalization == "fx":
            forward_signature = inspect.signature(self.model.forward).parameters
            if all(
                input_name in forward_signature for input_name in ["input_ids", "attention_mask", "past_key_values"]
            ):
                input_names = ["input_ids", "attention_mask", "past_key_values"]
            else:
                # TODO: implement
                raise ValueError("Not implemented")

            with torch.no_grad():
                model = symbolic_trace(self.model, input_names)
        else:
            model = self.model

        dtype = next(iter(model.parameters())).dtype

        # Insert standard MHA layers when performing fx based weight/act equalization to avoid dealing
        # with all the variability in HF implementations
        if quantization_config.replace_mha_with_quantizable:
            logger.info("Replace HF MHA with quantizable variants...")
            model = replace_mha_with_quantizable_layers(model, dtype)
            logger.info("Replacing done.")

        # Because accelerate is not compatible with FX, we keep two versions of the Model
        # one with FX-traced, the other one not.
        # Since weights are shared across the two, we can apply weight/activation equalization
        # by using one representation or the other based on needs.

        if quantization_config.apply_weight_equalization:
            logger.info("Applying weight equalization...")
            apply_weight_equalization(model)
            logger.info("Weight equalization applied.")

        if quantization_config.activations_equalization is not None:
            logger.info(
                f"Applying Activation Equalization {quantization_config.activations_equalization} (SmoothQuant)..."
            )
            apply_act_equalization(
                model, quantization_config.activations_equalization, calibration_dataloader, forward_call
            )
            logger.info("Activation equalization applied.")

        # We do not quantize embedding and last fully connected layer
        model = quantize_model(
            model,
            dtype=dtype,
            weight_quant_format="int",
            weight_quant_type="sym" if quantization_config.weights_symmetric else "asym",
            weight_bit_width=quantization_config.weights_bitwidth,
            weight_param_method=quantization_config.weights_param_method,
            weight_scale_precision=quantization_config.scale_precision,
            weight_quant_granularity=quantization_config.weights_quant_granularity,
            weight_group_size=quantization_config.weights_group_size,
            quantize_weight_zero_point=quantization_config.quantize_zero_point,
            input_bit_width=quantization_config.activations_bitwidth,
            input_quant_type="sym" if quantization_config.activations_symmetric else "asym",
            input_quant_format="int",
            input_param_method=quantization_config.activations_param_method,
            input_scale_precision=quantization_config.scale_precision,
            input_scale_type="static" if quantization_config.is_static else "dynamic",
            input_quant_granularity=quantization_config.activations_quant_granularity,
            input_group_size=quantization_config.activations_group_size,
            quantize_input_zero_point=quantization_config.quantize_zero_point,
            seqlen=quantization_config.seqlen,
        )

        # Perform a single inference pass to generate the correct state_dict
        if quantization_config.apply_gptq or quantization_config.is_static:
            with torch.no_grad():
                model(**calibration_dataloader[0])

        if quantization_config.apply_gptq:
            logger.info("Applying gptq...")
            apply_gptq(model, calibration_dataloader, forward_call)
            logger.info("GPTQ applied.")

        if quantization_config.activations_bitwidth is not None and quantization_config.is_static:
            logger.info("Applying activation calibration...")
            apply_calibration(model, calibration_dataloader, forward_call)
            logger.info("Activation calibration applied.")

        return model

    # TODO: move out of here
    def get_calibration_dataloader(
        self,
        tokenizer,
        dataset_name="c4",
        num_samples: int = 100,
        seqlen: int = 2048,
        seed: int = 0,
    ):
        if dataset_name == "c4":
            calibration_dataloader = get_c4(nsamples=num_samples, seed=seed, tokenizer=tokenizer, seqlen=seqlen)
        elif dataset_name == "wikitext2":
            calibration_dataloader = get_wikitext2(
                nsamples=num_samples, seqlen=seqlen, seed=seed, tokenizer=tokenizer, type=""
            )
        elif dataset_name == "wikitext2-raw":
            calibration_dataloader = get_wikitext2(
                nsamples=num_samples, seqlen=seqlen, seed=seed, tokenizer=tokenizer, type="raw"
            )

        return calibration_dataloader
