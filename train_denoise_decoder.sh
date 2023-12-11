python3 ./scripts/train_denoise_decoder.py \
 --pretrained_model_name_or_path="models/diffMamba-mini-sample" \
 --dataset_name="Gustavosta/Stable-Diffusion-Prompts" \
 --output_dir="models/diffmamba-mini-sample-trained" \
 --text_column Prompt \
 --train_batch_size 8 \
 --context_length 64 