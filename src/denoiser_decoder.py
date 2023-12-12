# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Classes to support Encoder-Decoder architectures"""


import gc
import inspect
import os
import tempfile
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, DenoiserDecoderConfig


logger = logging.get_logger(__name__)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class DenoiserDecoderModel(PreTrainedModel):

    config_class = DenoiserDecoderConfig
    base_model_prefix = "denoiser_decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        denoiser: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        scheduler = None
    ):
        if config is None and (denoiser is None or decoder is None):
            raise ValueError("Either a configuration or an denoiser and a decoder has to be provided.")
        if config is None:
            config = DenoiserDecoderConfig.from_denoiser_decoder_configs(denoiser.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.denoiser.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the denoiser's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.denoiser.hidden_size} for"
                    " `config.denoiser.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if denoiser is None:
            denoiser = AutoModel.from_config(config.denoiser)

        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.denoiser = denoiser
        self.decoder = decoder

        if self.denoiser.config.to_dict() != self.config.denoiser.to_dict():
            logger.warning(
                f"Config of the denoiser: {self.denoiser.__class__} is overwritten by shared denoiser config:"
                f" {self.config.denoiser}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.denoiser.config = self.config.denoiser
        self.decoder.config = self.config.decoder

        # denoiser outputs might need to be projected to different dimension for decoder
        if (
            self.denoiser.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.denoiser.config.hidden_size, self.decoder.config.hidden_size)

        if self.denoiser.get_output_embeddings() is not None:
            raise ValueError(
                f"The denoiser {self.denoiser} should not have a LM Head. Please use a model without LM Head"
            )

        decoder_signature = set(inspect.signature(self.decoder.forward).parameters.keys())
        if "encoder_hidden_states" not in decoder_signature:
            raise ValueError(
                "The selected decoder is not prepared for the encoder hidden states to be passed. Please see the "
                "following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350"
            )

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_denoiser_decoder:
            # tie denoiser and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_denoiser_decoder_weights(
                self.denoiser, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_denoiser(self):
        return self.denoiser

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.denoiser.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import DenoiserDecoderModel

        >>> model = DenoiserDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
        ```"""

        from_tf = kwargs.pop("from_tf", False)
        if from_tf:
            from transformers import TFDenoiserDecoderModel

            _tf_model = TFDenoiserDecoderModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            config = _tf_model.config

            # Using `tf_model` instead
            denoiser = _tf_model.denoiser.__class__(_tf_model.config.denoiser)
            decoder = _tf_model.decoder.__class__(_tf_model.config.decoder)
            # Make sure models are built
            denoiser(denoiser.dummy_inputs)
            decoder(decoder.dummy_inputs)

            # Get the variable correspondence between `_tf_model` and `denoiser` and `decoder`
            denoiser_variables = {}
            for v in denoiser.trainable_variables + denoiser.non_trainable_variables:
                denoiser_variables["/".join(v.name.split("/")[1:])] = v
            decoder_variables = {}
            for v in decoder.trainable_variables + decoder.non_trainable_variables:
                decoder_variables["/".join(v.name.split("/")[1:])] = v

            _denoiser_variables = {}
            for v in _tf_model.denoiser.trainable_variables + _tf_model.denoiser.non_trainable_variables:
                _denoiser_variables["/".join(v.name.split("/")[2:])] = v
            _decoder_variables = {}
            for v in _tf_model.decoder.trainable_variables + _tf_model.decoder.non_trainable_variables:
                _decoder_variables["/".join(v.name.split("/")[2:])] = v

            # assign weight values to `denoiser` and `decoder` from `_tf_model`
            for name, v in denoiser_variables.items():
                v.assign(_denoiser_variables[name])
            for name, v in decoder_variables.items():
                v.assign(_decoder_variables[name])

            tf_model = TFDenoiserDecoderModel(denoiser=denoiser, decoder=decoder)

            # Deal with `enc_to_dec_proj`
            if hasattr(_tf_model, "enc_to_dec_proj"):
                tf_model(tf_model.dummy_inputs)
                tf_model.enc_to_dec_proj.kernel.assign(_tf_model.enc_to_dec_proj.kernel)
                tf_model.enc_to_dec_proj.bias.assign(_tf_model.enc_to_dec_proj.bias)

            with tempfile.TemporaryDirectory() as tmpdirname:
                denoiser_dir = os.path.join(tmpdirname, "denoiser")
                decoder_dir = os.path.join(tmpdirname, "decoder")
                tf_model.denoiser.save_pretrained(denoiser_dir)
                tf_model.decoder.save_pretrained(decoder_dir)

                if hasattr(tf_model, "enc_to_dec_proj"):
                    enc_to_dec_proj_weight = torch.transpose(
                        torch.from_numpy(tf_model.enc_to_dec_proj.kernel.numpy()), 1, 0
                    )
                    enc_to_dec_proj_bias = torch.from_numpy(tf_model.enc_to_dec_proj.bias.numpy())

                del _tf_model
                del tf_model
                gc.collect()

                model = DenoiserDecoderModel.from_denoiser_decoder_pretrained(
                    denoiser_dir, decoder_dir, denoiser_from_tf=True, decoder_from_tf=True
                )
                # This is only for copying some specific attributes of this particular model.
                model.config = config

                if hasattr(model, "enc_to_dec_proj"):
                    model.enc_to_dec_proj.weight.data = enc_to_dec_proj_weight.contiguous()
                    model.enc_to_dec_proj.bias.data = enc_to_dec_proj_bias.contiguous()

                return model

        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for DenoiserDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_denoiser_decoder_pretrained(
        cls,
        denoiser_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:

        kwargs_denoiser = {
            argument[len("denoiser_") :]: value for argument, value in kwargs.items() if argument.startswith("denoiser_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove denoiser, decoder kwargs from kwargs
        for key in kwargs_denoiser.keys():
            del kwargs["denoiser_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the denoiser and decoder
        # The distinction between denoiser and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        denoiser = kwargs_denoiser.pop("model", None)
        if denoiser is None:
            if denoiser_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `denoiser_model` is not defined as an argument, a `denoiser_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_denoiser:
                denoiser_config, kwargs_denoiser = AutoConfig.from_pretrained(
                    denoiser_pretrained_model_name_or_path, **kwargs_denoiser, return_unused_kwargs=True
                )

                if denoiser_config.is_decoder is True or denoiser_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {denoiser_pretrained_model_name_or_path} as a denoiser model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    denoiser_config.is_decoder = False
                    denoiser_config.add_cross_attention = False

                kwargs_denoiser["config"] = denoiser_config

            denoiser = AutoModel.from_pretrained(denoiser_pretrained_model_name_or_path, *model_args, **kwargs_denoiser)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_denoiser_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_denoiser_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = DenoiserDecoderConfig.from_denoiser_decoder_configs(denoiser.config, decoder.config, **kwargs)
        return cls(denoiser=denoiser, decoder=decoder, config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        denoiser_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import DenoiserDecoderModel, BertTokenizer
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = DenoiserDecoderModel.from_denoiser_decoder_pretrained(
        ...     "bert-base-uncased", "bert-base-uncased"
        ... )  # initialize Bert2Bert from pre-trained checkpoints

        >>> # training
        >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
        >>> model.config.pad_token_id = tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
        >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss, logits = outputs.loss, outputs.logits

        >>> # save and load from pretrained
        >>> model.save_pretrained("bert2bert")
        >>> model = DenoiserDecoderModel.from_pretrained("bert2bert")

        >>> # generation
        >>> generated = model.generate(input_ids)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_denoiser = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if denoiser_outputs is None:
            denoiser_outputs = self.denoiser(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_denoiser,
            )
        elif isinstance(denoiser_outputs, tuple):
            denoiser_outputs = BaseModelOutput(*denoiser_outputs)

        encoder_hidden_states = denoiser_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.denoiser.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            if decoder_attention_mask is None:
                decoder_attention_mask = decoder_input_ids.new_tensor(decoder_input_ids != self.config.pad_token_id)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + denoiser_outputs
            else:
                return decoder_outputs + denoiser_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=denoiser_outputs.last_hidden_state,
            encoder_hidden_states=denoiser_outputs.hidden_states,
            encoder_attentions=denoiser_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, denoiser_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "denoiser_outputs": denoiser_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the DenoiserDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.denoiser.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)