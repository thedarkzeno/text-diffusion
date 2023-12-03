from modeling_diffbert_sample import DiffBertForDiffusion, DiffBertConfig
from transformers import AutoTokenizer
from diffusers import DDIMScheduler


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


scheduler = DDIMScheduler(prediction_type="sample", num_train_timesteps = 2000)
config = DiffBertConfig(hidden_size=768, num_hidden_layers=6, intermediate_size=768, vocab_size=tokenizer.vocab_size, timesteps=2000)

model = DiffBertForDiffusion(config)

model.save_pretrained("diffbert-mini-sample")
tokenizer.save_pretrained("diffbert-mini-sample")
scheduler.save_pretrained("diffbert-mini-sample")