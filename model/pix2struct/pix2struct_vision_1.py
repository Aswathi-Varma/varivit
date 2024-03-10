# coding=utf-8
# Copyright 2023 The HuggingFace Inc. & Google team. All rights reserved.
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
""" Pix2Struct modeling file"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from model.pix2struct.activations import ACT2FN
import torch.nn.functional as F

from model.pix2struct.configuration_pix2struct import Pix2StructVisionConfig

# Adapted from transformers.models.t5.modeling_t5.T5LayerNorm with T5->Pix2Struct

class Pix2StructLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class Pix2StructVisionEmbeddings(nn.Module):
    r"""
    Construct the embeddings from patch. In `Pix2Struct` the input is different from classic Vision-transformer models.
    Here the input is a sequence of `seq_len` flattened patches that also combines padding patches (tokens). Each patch
    is represented by a vector of `hidden_size` values.
    """

    def __init__(self, config: Pix2StructVisionConfig) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)

        self.depth_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.row_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.column_embedder = nn.Embedding(config.seq_len, config.hidden_size)
    
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        device = flattened_patches.device
        # the row and column indices are stored in the first and second position of the flattened_patches
        # flattened_patches: `batch_size`, `seq_len`, `hidden_size` + 2
        depth_indices = flattened_patches[:, :, 0].long()
        row_indices = flattened_patches[:, :, 1].long()
        col_indices = flattened_patches[:, :, 2].long()

        flattened_patches = flattened_patches[:, :, 3:]
        
        embeddings = self.patch_projection(flattened_patches)
        # Count the number of non-zero elements
        num_non_zeros = torch.nonzero(embeddings).size(0)


        depth_embeddings = self.depth_embedder(depth_indices.to(device))
        row_embeddings = self.row_embedder(row_indices.to(device))
        col_embeddings = self.column_embedder(col_indices.to(device))
        
        # sum all embeddings together
        embeddings = embeddings + depth_embeddings + row_embeddings + col_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))
    
    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta, 1e-13)


class Pix2StructVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.hidden_size // config.num_attention_heads
        self.n_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        #self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        Self-attention block
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]
        def to_projection_shape(states):
            """projection"""
            return states.contiguous().view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # get query states
        # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = to_projection_shape(self.query(hidden_states))

        # get key/value states
        key_states = to_projection_shape(self.key(hidden_states))
        value_states = to_projection_shape(self.value(hidden_states))

        # compute scores
        # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            position_bias = torch.zeros(
                (1, self.n_heads, seq_length, seq_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True

            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), device=scores.device, dtype=scores.dtype)

            if attention_mask.dim() == 2:
                position_bias = position_bias + attention_mask[:, None, None, :].to(position_bias.device)
            else:
                # (batch_size, n_heads, seq_length, key_length)
                position_bias = position_bias + attention_mask.to(position_bias.device)
            position_bias = 1 - position_bias

        position_bias_masked = position_bias.masked_fill(position_bias == 1, torch.finfo(scores.dtype).min)
        scores += position_bias_masked
        scores = torch.max(scores, torch.tensor(torch.finfo(scores.dtype).min))

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        # (batch_size, seq_length, dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        attn_output = self.output(attn_output)

        outputs = (attn_output,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5DenseGatedActDense->Pix2StructVisionMlp,T5Config->Pix2StructVisionConfig,config.d_model->config.hidden_size,dropout_rate->dropout_rate
class Pix2StructVisionMlp(nn.Module):
    def __init__(self, config: Pix2StructVisionConfig):
        super().__init__()
        mlp_hidden_dim = int(config.hidden_size * 4)

        self.wi_0 = nn.Linear(config.hidden_size, mlp_hidden_dim, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, mlp_hidden_dim, bias=False)
        self.wo = nn.Linear(mlp_hidden_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class Pix2StructVisionLayer(nn.Module):
    def __init__(self, config: Pix2StructVisionConfig) -> None:
        super().__init__()
        
        self.seq_len_dim = 1
        self.attention = Pix2StructVisionAttention(config)
        self.mlp = Pix2StructVisionMlp(config)
        self.pre_mlp_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_attention_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        residual = hidden_states

        # in Pix2StructVision, layernorm is applied before self-attention
        hidden_states = self.pre_attention_layer_norm(hidden_states)

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + residual

        # in Pix2StructVision, layernorm is also applied after self-attention
        layer_output = self.pre_mlp_layer_norm(hidden_states)
        layer_output = self.mlp(layer_output) + hidden_states  # second residual connection

        outputs = (layer_output,) + outputs

        return outputs


class Pix2StructVisionEncoder(nn.Module):
    def __init__(self, config: Pix2StructVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Pix2StructVisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states

class Pix2StructVisionModel(nn.Module):
    config_class = Pix2StructVisionConfig
    main_input_name = "flattened_patches"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Pix2StructVisionLayer"]

    def __init__(self, config: Pix2StructVisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = Pix2StructVisionEmbeddings(config)
        self.encoder = Pix2StructVisionEncoder(config)
        # Classifier head(s)
        self.head = nn.Linear(config.hidden_size, config.num_classes) if config.num_classes > 0 else nn.Identity()
        self.layernorm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.apply(self._init_weights)
    
    def get_input_embeddings(self):
        return self.embeddings.patch_projection

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, Pix2StructLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Pix2StructLayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        flattened_patches: Optional[torch.Tensor] = None,
        attention_mask = None,
    ):
        
        if flattened_patches is None:
            raise ValueError("You have to specify flattened_patches")

        if attention_mask is None:
            # check where `flattened_patches` is not 0
            attention_mask = (flattened_patches.sum(dim=-1) != 0).float()
        embedding_output = self.embeddings(flattened_patches)
        encoder_output = self.encoder(embedding_output)

        #global pooling
        sequence_output = encoder_output.mean(dim=1)
        sequence_output = self.layernorm(sequence_output)

        #classifier head
        outcome = self.head(sequence_output)

        return outcome