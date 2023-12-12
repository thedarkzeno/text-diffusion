python3 train_seq2seq_completion.py \
 --pretrained_model_name_or_path="models/diffMamba-mini-sample" \
 --train_file="../roberta/data/brwac-train.txt" \
 --output_dir="Gustavosta/Stable-Diffusion-Prompts" \
 --text_column text \
 --train_batch_size 8 \
 --context_length 128 