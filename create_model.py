# from modeling_diffbert_sample import DiffBertForDiffusion, DiffBertConfig
from src.modeling_diffmamba import DiffMambaForDiffusionLM, DiffMambaConfig
from transformers import AutoTokenizer
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

timesteps = 1000
scheduler = DDIMScheduler(
  beta_end = 0.012,
  beta_schedule = "scaled_linear",
  beta_start = 0.00085,
  clip_sample = False,
#   skip_prk_steps = True,
  set_alpha_to_one = False,
  steps_offset = 1,
#   interpolation_type = "linear",
  prediction_type ="sample", 
  num_train_timesteps = timesteps
)


config = DiffMambaConfig(
        hidden_size=768,
        num_hidden_layers=28,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=tokenizer.vocab_size, 
        timesteps=timesteps, 
        torch_dtype=torch.float16
    )

model = DiffMambaForDiffusionLM(config)

model.save_pretrained("models/diffMamba-mini-sample")
tokenizer.save_pretrained("models/diffMamba-mini-sample")
# scheduler.save_pretrained("models/diffMamba-mini-sample")