from typing import Callable
from torch import manual_seed
import numpy as np
import random
import torch.nn as nn

def seed_everything(random_state):
    manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    return None

def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x

def get_ff_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Gets the feedforward layer of a model within the transformer block
    `model`: torch.nn.Module
      a torch.nn.Module
    `layer_idx`: int
      which transformer layer to access
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)
    assert layer_idx < len(
        transformer_layers
    ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    return ff_layer

def register_hook(
    model: nn.Module,
    layer_idx: int,
    f: Callable,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Registers a forward hook in a pytorch transformer model that applies some function, f, to the intermediate
    activations of the transformer model.

    specify how to access the transformer layers (which are expected to be indexable - i.e a ModuleList) with transformer_layers_attr
    and how to access the ff layer with ff_attrs

    `model`: torch.nn.Module
      a torch.nn.Module
    `layer_idx`: int
      which transformer layer to access
    `f`: Callable
      a callable function that takes in the intermediate activations
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    ff_layer = get_ff_layer(
        model,
        layer_idx,
        transformer_layers_attr=transformer_layers_attr,
        ff_attrs=ff_attrs,
    )

    def hook_fn(m, i, o):
        f(o)

    return ff_layer.register_forward_hook(hook_fn)

def set_attribute_recursive(x: nn.Module, attributes: "str", new_attribute: nn.Module):
    """
    Given a list of period-separated attributes - set the final attribute in that list to the new value
    i.e set_attribute_recursive(model, 'transformer.encoder.layer', NewLayer)
        should set the final attribute of model.transformer.encoder.layer to NewLayer
    """
    for attr in attributes.split(".")[:-1]:
        x = getattr(x, attr)
    setattr(x, attributes.split(".")[-1], new_attribute)