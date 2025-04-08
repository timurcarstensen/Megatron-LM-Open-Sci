ACCOUNT={account}
PARTITION={partition}
CONTAINER_IMAGE={container_image}
TRAIN_LOGS_DIR_OR_PATH={train_logs}
CONVERT_LOGS_DIR={convert_logs_dir}
mkdir -p $CONVERT_LOGS_DIR

OPENSCI_MEGATRON_PATH={opensci_megatron_path}
# MEGATRON_PATH={megatron_path}
OPEN_SCI_HF_PATH={open_sci_hf_path}
SAVE_CHECKPOINTS_DIR={save_checkpoints_dir}


SCRIPT={opensci_megatron_path}/scripts/ckpt/convert_full/converter.py

CMD="python3 $SCRIPT \
    --container_image $CONTAINER_IMAGE \
    --train_logs_dir_or_path $TRAIN_LOGS_DIR_OR_PATH \
    --convert_logs_dir $CONVERT_LOGS_DIR \
    --save_checkpoints_dir $SAVE_CHECKPOINTS_DIR \
    --account $ACCOUNT \
    --partition $PARTITION \
    --open_sci_hf_path $OPEN_SCI_HF_PATH \
    --opensci_megatron_path $OPENSCI_MEGATRON_PATH \
    --iters {iters_to_convert} \ 
    --num_layers {num_layers} \
    --num_attn_heads {num_attn_heads} \
    --ffn_hidden_size {ffn_hidden_size} \
    --max_seq_length {max_seq_length} "

echo $CMD
$CMD