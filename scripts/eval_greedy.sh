DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# MODEL_PATH=/SSD/huggingface/meta-llama
# # MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# # MODEL_NAME=Llama-2-70b-hf
# CONFIG=config/llama.json
# DTYPE=float16

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B
# MODEL_NAME=Llama-3.1-70B
CONFIG=config/llama.json
DTYPE=bfloat16

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B
# MODEL_NAME=Qwen2.5-14B
# # MODEL_NAME=Qwen2.5-32B
# MODEL_NAME=Qwen2.5-72B
# DTYPE=bfloat16
# CONFIG=config/qwen2.json

Q_BITS="2 3 4"
Q_BITS_TEXT="234"

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

# GREEDY_SEARCH_RESULT=/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_hqq_24bits_loss_desc_1axis_128gs_jsd.csv
# LAST_LINEAR=

# GREEDY_SEARCH_RESULT=/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-13b-hf_hqq_24bits_loss_desc_1axis_128gs_jsd.csv
# LAST_LINEAR=21.self_attn.o_proj # 3.5
# # LAST_LINEAR=22.mlp.up_proj # 3.0
# # LAST_LINEAR=36.mlp.gate_proj # 2.5

GREEDY_SEARCH_RESULT=/NAS/SJ/nsgaquant/csv/greedy_search/Llama-3.1-8B_hqq_24bits_loss_desc_1axis_128gs_jsd.csv
LAST_LINEAR=7.mlp.gate_proj # 3.5
LAST_LINEAR=8.mlp.up_proj # 3.0
LAST_LINEAR=31.mlp.gate_proj # 2.5

# GREEDY_SEARCH_RESULT=/NAS/SJ/nsgaquant/csv/greedy_search/Qwen2.5-7B_hqq_24bits_loss_desc_1axis_128gs_jsd.csv
# LAST_LINEAR=5.mlp.down_proj # 3.5
# LAST_LINEAR=25.mlp.down_proj # 3.0
# LAST_LINEAR=22.mlp.gate_proj # 2.5


DATASETS="wikitext2 c4"

N_PROC=1
# N_PROC=2

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} \
eval_greedy.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--quant_model_paths ${QMODEL_PATHS} \
--quant_model_bits ${Q_BITS} \
--greedy_search_result ${GREEDY_SEARCH_RESULT} \
--last_linear ${LAST_LINEAR} \
--datasets ${DATASETS} \
--method ${METHOD} \
--zeroshot \
--tasks ${TASKS}

# --outlier_path ${OUTLIER_PATH} \
# --only_front \
