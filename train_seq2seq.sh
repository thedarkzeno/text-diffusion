python3 train_seq2seq.py \
 --pretrained_model_name_or_path="models/diffMamba-mini-sample" \
 --dataset_name="Electrofried/promptmaster-data" \
 --output_dir="models/diffmamba-mini-sample-trained" \
 --text_column data2 \
 --instruction_column data1