# from modeling_diffbert_sample import DiffBertForDiffusion, DiffBertConfig
from src.denoisers.modeling_diffmamba import DiffMambaForDiffusionLM, DiffMambaConfig
from transformers import AutoTokenizer, BertLMHeadModel, BertConfig
from src.schedulers.ddpm import DDPMScheduler
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

timesteps = 1200
scheduler = DDPMScheduler(
  beta_schedule = "sqrt",
  prediction_type ="sample", 
  num_train_timesteps = timesteps
)


config = DiffMambaConfig(
        hidden_size=768,
        num_hidden_layers=20,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=tokenizer.vocab_size, 
        timesteps=timesteps, 
        torch_dtype=torch.float16
    )

decoder_config = BertConfig(
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=tokenizer.vocab_size, 
        is_decoder=True,
        add_cross_attention=True,
        torch_dtype=torch.float16
    )

model = DiffMambaForDiffusionLM(config)
decoder = BertLMHeadModel(decoder_config)

model.save_pretrained("models/diffMamba-mini-sample/denoiser")
tokenizer.save_pretrained("models/diffMamba-mini-sample/tokenizer")
scheduler.save_pretrained("models/diffMamba-mini-sample/scheduler")
decoder.save_pretrained("models/diffMamba-mini-sample/decoder")