DEVICES=${1}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf
# MODEL_NAME=Meta-Llama-3-8B

# MODEL=facebook/opt-6.7b
# MODEL=facebook/opt-13b
# MODEL=facebook/opt-30b
# MODEL=facebook/opt-66b

METHOD=layer_prune
LOSS_FUNC=jsd
N_SAMPLE=128
CONFIG=config/llama.json

LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_${LOSS_FUNC}_bf16.csv
# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_${LOSS_FUNC}.csv
PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_ppl_${LOSS_FUNC}.csv

N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} layer_sensitivity.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--method ${METHOD} \
--n_sample ${N_SAMPLE} \
--ppl_csv_file ${PPL_CSV_FILE} \
--config ${CONFIG} \
--loss_csv_file ${LOSS_CSV_FILE} \
--loss_func ${LOSS_FUNC}
# --eval_ppl

# CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID 
# --eval_zeroshot \
# CUDA_DEVICE_ORDER=PCI_BUS_ID 