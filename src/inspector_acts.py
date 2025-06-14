import collections
import math
from functools import partial
from typing import Callable, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

# from patch.py
import copy
from patch import *

# from setup.py
from transformers import (
    BertTokenizer, AutoModelForSequenceClassification
)


class BERTInspectorBase:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "bert",
        device: str = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.baseline_activations = None

        self.transformer_layers_attr = "bert.encoder.layer"
        self.input_ff_attr = "intermediate"
        self.output_ff_attr = "output.dense.weight"

    ## UTILITY FUNCTIONS
    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _prepare_inputs(self, prompt, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        return encoded_input

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        return self.model.config.intermediate_size


    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, token_idx: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `token_idx`: int
            the position at which to get the activations 
        """

        def get_activations(model, layer_idx, token_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, token_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, token_idx=token_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations
    
    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        if activations.dim() == 2:
            tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
            return (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None]
            )
        elif activations.dim() == 3:
            tiled_activations = einops.repeat(activations, "b m d -> (r b) m d", r=steps)
            return (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None, None]
            )
        else:
            raise Exception(f"Bad!! The dim of Activation is {activations.dim()}")

    ## SCORE FOR ALL LAYERS
    def get_scores(
        self,
        prompt: str,
        label: str,
        token_indexes: list,
        batch_size: int = 5,
        ig_steps: int = 5,
        attribution_method: str = "activations",
        pbar: bool = True,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `label`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `ig_steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """

        scores = []
        encoded_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        for layer_idx in tqdm(
                range(self.n_layers()),
                desc="Getting attribution scores for each layer...",
                disable=not pbar,
                leave=False,
                position=1
            ):
            layer_scores = self.get_scores_for_layer(
                prompt,
                label,
                token_indexes,
                encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                ig_steps=ig_steps,
                attribution_method=attribution_method,
            )
            scores.append(layer_scores)
        scores = [score.to(self.device) for score in scores]
        return torch.stack(scores)


    def get_scores_for_layer(
            self,
            prompt: str,
            target_label: int,
            token_indexes: list,
            layer_idx: int,
            batch_size: int = 5,
            ig_steps: int = 5,
            encoded_input: Optional[int] = None,
            attribution_method: str = "integrated_grads",
        ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert ig_steps % batch_size == 0
        n_batches = ig_steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        if encoded_input is None:
            print("No encoded inputs given, computing now.")
            encoded_input = self._prepare_inputs(
                prompt, encoded_input
            )

        activations_all_toks = []
        
        for token_idx in token_indexes:
            (
                baseline_outputs,
                baseline_activations,
            ) = self.get_baseline_with_activations(
                encoded_input, layer_idx, token_idx
            )  #baseline activations dim = 1 x 3072 (intermediate layer size)

            # # WHY DOES THIS WORK WELL FOR 1 TOPIC???
            # (
            #     baseline_outputs,
            #     baseline_activations,
            # ) = self.get_baseline_with_activations(
            #     encoded_input, layer_idx, 0
            # )  #baseline activations dim = 1 x 3072 (intermediate layer size)
            
            
            activations_all_toks.append(baseline_activations)
        activations_all_toks = torch.stack(activations_all_toks, dim=0)
        return activations_all_toks
    

def initialize_model_and_tokenizer(model_name: str, torch_dtype='auto'):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", clean_up_tokenization_spaces=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    return model, tokenizer

def model_type(model_name: str):
    if "bert" in model_name:
        return "bert"
    else:
        raise ValueError("Model {model_name} not supported")

def load_model(model_name_or_path, device=None):
    model, tokenizer = initialize_model_and_tokenizer(model_name_or_path)
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    kn = BERTInspectorBase(model, tokenizer, model_type=model_type(model_name_or_path))
    return kn