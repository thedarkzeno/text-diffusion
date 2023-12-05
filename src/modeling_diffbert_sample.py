# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BERT model."""


import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    is_flash_attn_2_available,
    replace_return_docstrings,
)
from src.configuration_diffbert import DiffBertConfig
import numpy as np

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
@dataclass
class DiffusionLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class DiffBertSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)

class DiffBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # self.RMSNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # self.RMSNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # self.register_buffer(
        #     "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        # )
        # self.register_buffer(
        #     "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        # )

    def apply_position(self, input_ids: torch.LongTensor, past_key_values_length: int = 0):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        position_embeddings = self.position_embeddings(position_ids)
        return input_ids + position_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # if input_ids is not None:
        #     input_shape = input_ids.size()
        # else:
        #     input_shape = input_embeds.size()[:-1]

        # seq_length = input_shape[1]

        # if position_ids is None:
        #     position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # # issue #5664
        # if token_type_ids is None:
        #     if hasattr(self, "token_type_ids"):
        #         buffered_token_type_ids = self.token_type_ids[:, :seq_length]
        #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
        #         token_type_ids = buffered_token_type_ids_expanded
        #     else:
        #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if input_embeds is None:
            input_embeds = self.word_embeddings(input_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        

        # embeddings = input_embeds + token_type_embeddings# + time_embeddings
        # if self.position_embedding_type == "absolute":
        #     position_embeddings = self.position_embeddings(position_ids)
        #     embeddings += position_embeddings
        # embeddings = self.RMSNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return input_embeds


class DiffBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.rotary_value = config.rotary_value

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            if sinusoidal_pos is not None:
                if self.rotary_value:
                    query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer, value_layer
                    )
                else:
                    query_layer, key_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer
                    )
            if past_key_value is not None:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RoFormerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer

# class DiffBertSelfFlashAttention2(DiffBertSelfAttention):
#     def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(new_x_shape)
#         return x
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         sinusoidal_pos=None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.FloatTensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.Tensor]:
        
#         mixed_query_layer = self.query(hidden_states)
#         batch_size, query_length, hidden_size = mixed_query_layer.shape
        
#         is_cross_attention = encoder_hidden_states is not None

#         if is_cross_attention and past_key_value is not None:
#             # reuse k,v, cross_attentions
#             key_layer = past_key_value[0]
#             value_layer = past_key_value[1]
#             attention_mask = encoder_attention_mask
#         elif is_cross_attention:
#             key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
#             value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
#             attention_mask = encoder_attention_mask
#         elif past_key_value is not None:
#             key_layer = self.transpose_for_scores(self.key(hidden_states))
#             value_layer = self.transpose_for_scores(self.value(hidden_states))
#             key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
#             value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
#         else:
#             key_layer = self.transpose_for_scores(self.key(hidden_states))
#             value_layer = self.transpose_for_scores(self.value(hidden_states))

#         query_layer = self.transpose_for_scores(mixed_query_layer)

#         use_cache = past_key_value is not None
#         if self.is_decoder:
#             past_key_value = (key_layer, value_layer)
        
#         if attention_mask is None:
#             attention_scores = flash_attn_func(
#                 query_layer, key_layer, value_layer, self.dropout.p, softmax_scale=None, causal=self.is_decoder
#             )
#         else:
#             query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
#                 query_layer, key_layer, value_layer, attention_mask, query_length
#             )
            
#             cu_seqlens_q, cu_seqlens_k = cu_seq_lens
#             max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            
#             attn_output_unpad = flash_attn_varlen_func(
#                 query_states,
#                 key_states,
#                 value_states,
#                 cu_seqlens_q=cu_seqlens_q,
#                 cu_seqlens_k=cu_seqlens_k,
#                 max_seqlen_q=max_seqlen_in_batch_q,
#                 max_seqlen_k=max_seqlen_in_batch_k,
#                 dropout_p=self.dropout.p,
#                 softmax_scale=None,
#                 causal=self.is_decoder,
#             )
            
#             attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        
#         attn_output = attn_output.reshape(batch_size, query_length, hidden_size).contiguous()
        
#         if output_attentions:
#             raise NotImplementedError("output_attentions is not implemented")
        
#         outputs = (attn_output, None)
#         if self.is_decoder:
#             outputs = outputs + (past_key_value,)
#         return outputs
            
        
#     def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
#         # print(attention_mask)
#         # if attention_mask == None:
#         #     shape = query_layer.shape
#         attention_mask = torch.ones_like(attention_mask).to(query_layer.device)
#         indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
#         print(cu_seqlens_k, cu_seqlens_k)
#         batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

#         key_layer = index_first_axis(
#             key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
#         )
#         value_layer = index_first_axis(
#             value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
#         )
#         if query_length == kv_seq_len:
#             query_layer = index_first_axis(
#                 query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads, head_dim), indices_k
#             )
#             cu_seqlens_q = cu_seqlens_k
#             max_seqlen_in_batch_q = max_seqlen_in_batch_k
#             indices_q = indices_k
#         elif query_length == 1:
#             max_seqlen_in_batch_q = 1
#             cu_seqlens_q = torch.arange(
#                 batch_size + 1, dtype=torch.int32, device=query_layer.device
#             )  # There is a memcpy here, that is very bad.
#             indices_q = cu_seqlens_q[:-1]
#             query_layer = query_layer.squeeze(1)
#         else:
#             # The -q_len: slice assumes left padding.
#             attention_mask = attention_mask[:, -query_length:]
#             query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

#         return (
#             query_layer,
#             key_layer,
#             value_layer,
#             indices_q,
#             (cu_seqlens_q, cu_seqlens_k),
#             (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
#         )

class DiffBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.RMSNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)
        return hidden_states


class DiffBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.self = (
        #     DiffBertSelfAttention(config)
        #     if not getattr(config, "_flash_attn_2_enabled", False)
        #     else DiffBertSelfFlashAttention2(config)
        # )#DiffBertSelfAttention(config)
        self.self = DiffBertSelfAttention(config)
        self.output = DiffBertSelfOutput(config)
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # End Copy
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class DiffBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.config.intermediate_size, self.config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return hidden_states


class DiffBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.RMSNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)
        return hidden_states


class DiffBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = DiffBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = DiffBertAttention(config, position_embedding_type="absolute")
        self.intermediate = DiffBertIntermediate(config)
        self.output = DiffBertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        sinusoidal_pos: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                sinusoidal_pos,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class DiffBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_positions = DiffBertSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads
        )
        self.layer = nn.ModuleList([DiffBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1], 0)[None, None, :, :]
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class DiffBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DiffBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.RMSNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.RMSNorm(hidden_states)
        return hidden_states


class DiffBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DiffBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DiffBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DiffBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class DiffBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2, bias=False)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class DiffBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DiffBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2, bias=False)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class DiffBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DiffBertConfig
    base_model_prefix = "diffbert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # if isinstance(module, nn.Linear):
        #     # Slightly different from the TF version which uses truncated_normal for initialization
        #     # cf https://github.com/pytorch/pytorch/pull/5617
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     if module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
        # elif isinstance(module, RMSNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        return


@add_start_docstrings(
    "The bare DiffBert Model transformer outputting raw hidden-states without any specific head on top.",
)
class DiffBertModel(DiffBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = DiffBertEmbeddings(config)
        #added position + timesteps
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.input_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size//2, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size//2, config.hidden_size, bias=False),
        )#nn.Embedding(config.timesteps, config.hidden_size)

        # self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # self.register_buffer(
        #     "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        # )
        self.RMSNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.encoder = DiffBertEncoder(config)

        self.pooler = DiffBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        input_embeds,
        timesteps,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        
        if input_embeds is not None:
            input_shape = input_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or input_embeds")

        batch_size, seq_length = input_shape
        device = input_embeds.device if input_embeds is not None else input_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # embedding_output = self.embeddings(
        #     position_ids=position_ids,
        #     token_type_ids=token_type_ids,
        #     input_embeds=input_embeds,
        #     timesteps=timesteps,
        #     past_key_values_length=past_key_values_length,
        # )
        n_seq = input_embeds.size(1)
        # repeated_tensor = timesteps.unsqueeze(1).repeat(1, n_seq)
        # if position_ids is None:
        # position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        # position_embeddings = self.position_embeddings(position_ids)
        # embeddings += position_embeddings
        # time_embeddings = self.time_embedding(repeated_tensor)
        input_embeds = self.input_proj(input_embeds)
        time_embeddings = timestep_embedding(timesteps, self.config.hidden_size).unsqueeze(1).repeat(1, n_seq, 1)
        time_embeddings = self.time_embedding(time_embeddings.to(input_embeds.dtype))
        input_embeds += time_embeddings #+ position_embeddings
        input_embeds = self.RMSNorm(input_embeds)

        encoder_outputs = self.encoder(
            input_embeds,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )



@add_start_docstrings("""DiffBert Model with a `language modeling` head on top.""")
class DiffBertForDiffusion(DiffBertPreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `DiffBertForDiffusion` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = DiffBertModel(config, add_pooling_layer=False)

        self.out_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        )
        self.cls = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_output_embeddings(self):
        return #.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        return
        # self.cls.predictions.decoder = new_embeddings
    
    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value
    
    def apply_embeddings(self, input_ids, token_type_ids=None, position_ids=None):
        return self.bert.embeddings(input_ids,
        token_type_ids,
        position_ids)
    
    def apply_lm_head(self, latents, labels=None):
        prediction_scores = self.cls(latents)
        masked_lm_loss=None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return DiffusionLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            last_hidden_state=latents,
            attentions=None,
        )


    def forward(
        self,
        input_embeds,
        timesteps,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            input_embeds=input_embeds,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.out_proj(sequence_output)
        # print(sequence_output.shape)
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return DiffusionLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            last_hidden_state=sequence_output,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

