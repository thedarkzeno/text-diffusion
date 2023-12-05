# from modeling_diffbert_sample import DiffBertForDiffusion, DiffBertConfig
from modeling_diffllama import DiffLlamaForDiffusionLM, DiffLlamaConfig
from transformers import AutoTokenizer
from diffusers import DDIMScheduler
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


scheduler = DDIMScheduler(prediction_type="sample", num_train_timesteps = 2000)
config = DiffLlamaConfig(num_hidden_layers=6, vocab_size=tokenizer.vocab_size, timesteps=2000, torch_dtype=torch.float16)

model = DiffLlamaForDiffusionLM(config)

model.save_pretrained("diffllama-mini-sample")
tokenizer.save_pretrained("diffllama-mini-sample")
scheduler.save_pretrained("diffllama-mini-sample")