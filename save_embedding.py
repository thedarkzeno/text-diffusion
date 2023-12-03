from transformers import BertForMaskedLM
import torch

model = BertForMaskedLM.from_pretrained("neuralmind/bert-base-portuguese-cased")
embedding = model.bert.embeddings.word_embeddings
torch.save(embedding.state_dict(), 'diffbert-mini/embedding_weights.bin')