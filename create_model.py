from modeling_diffbert import DiffBertForDiffusion, DiffBertConfig
from transformers import AutoTokenizer
from diffusers import DDIMScheduler



config = DiffBertConfig(hidden_size=768, num_hidden_layers=2, intermediate_size=768, vocab_size=29794)

model = DiffBertForDiffusion(config)
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
scheduler = DDIMScheduler()

model.save_pretrained("diffbert-mini")
tokenizer.save_pretrained("diffbert-mini")
scheduler.save_pretrained("diffbert-mini")