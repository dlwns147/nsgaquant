DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf
CONFIG=config/llama.json
DTYPE=float16

# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-3.1-8B
# # MODEL_NAME=Llama-3.1-70B
# CONFIG=config/llama.json
# DTYPE=bfloat16

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B
# MODEL_NAME=Qwen2.5-14B
# # MODEL_NAME=Qwen2.5-32B
# MODEL_NAME=Qwen2.5-72B
# DTYPE=bfloat16
# CONFIG=config/qwen2.json

Q_BITS="2 4"
Q_BITS_TEXT="24"

# METHOD="hqq layer_prune"
# METHOD_TEXT="hqq_layer_prune"

METHOD=hqq
METHOD_TEXT=hqq

# METHOD=awq
# METHOD_TEXT=awq

# METHOD="awq layer_prune"
# METHOD_TEXT=awq_layer_prune

GROUP_SIZE=128
AXIS=1
QSCALE=false
QZERO=false


QMODEL_PATHS_LIST=()
for B in ${Q_BITS}
do
    # QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

N_OUTLIER=32
OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq openbookqa social_iqa"

LINEAR_SENSITIVITY=/NAS/SJ/nsgaquant/csv/sensitivity/Llama-2-7b-hf_hqq_loss_24_1axis_128gs_false_qs_false_qz_jsd.csv
# LINEAR_SENSITIVITY=/NAS/SJ/nsgaquant/csv/sensitivity/Llama-2-13b-hf_hqq_loss_24_1axis_128gs_false_qs_false_qz_jsd.csv
# LINEAR_SENSITIVITY=/NAS/SJ/nsgaquant/csv/sensitivity/Llama-3.1-8B_hqq_loss_24_1axis_128gs_false_qs_false_qz_jsd.csv

TARGET_BITS=2.5
# TARGET_BITS=3.0
# TARGET_BITS=3.5

# DATASETS="wikitext2 c4"
DATASETS="wikitext2"

N_PROC=1
# N_PROC=2

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} \
eval_naive.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--quant_model_paths ${QMODEL_PATHS} \
--quant_model_bits ${Q_BITS} \
--linear_sensitivity ${LINEAR_SENSITIVITY} \
--target_bit ${TARGET_BITS} \
--datasets ${DATASETS} \
--method ${METHOD}
# --zeroshot \
# --tasks ${TASKS}

# --outlier_path ${OUTLIER_PATH} \
# --only_front \
