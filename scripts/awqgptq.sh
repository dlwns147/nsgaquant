DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# MODEL_PATH=/SSD/huggingface/meta-llama
# # MODEL_NAME=Llama-2-7b-hf
# # MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf
# CONFIG=config/llama.json
# DTYPE=float16

# MODEL_PATH=/SSD/huggingface/meta-llama
# # MODEL_NAME=Llama-3.1-8B
# MODEL_NAME=Llama-3.1-70B
# # MODEL_NAME=Llama-3.1-8B-Instruct
# CONFIG=config/llama.json
# DTYPE=bfloat16

# MODEL_PATH=/SSD/huggingface/Qwen
# # MODEL_NAME=Qwen2.5-7B
# # MODEL_NAME=Qwen2.5-14B
# # MODEL_NAME=Qwen2.5-32B
# MODEL_NAME=Qwen2.5-72B
# CONFIG=config/qwen2.json
# DTYPE=bfloat16

MODEL_PATH=/SSD/huggingface/mistralai
MODEL_NAME=Mistral-7B-v0.3
DTYPE=bfloat16
CONFIG=config/mistral.json

Q_BITS="2 3 4"
Q_BITS_TEXT="234"
AXIS=1
GROUP_SIZE=128

QMODEL_PATHS_LIST=()
for B in ${Q_BITS}
do
    # QMODEL_PATHS+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")


# METHOD=hqq
METHOD=awq
# METHOD=gptq

COMP_OBJ="bits"
COMP_OBJ_TEXT=bits


TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq"
# BATCH_SIZE=16
# BATCH_SIZE=32
BATCH_SIZE=64

N=1
DATASETS="wikitext2 c4"

GROUP_SIZE=128
# GROUP_SIZE=-1

SAVE=save/result/${TODAY}_${MODEL_NAME}_${COMP_OBJ}_${METHOD}_${BITS}

TARGET_COMP_OBJ=bits
# TARGET_BITS_LIST=(2 3 4)
BITS=3
# BITS=4
# BITS=16
N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} awqgptq.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--quant_model_paths ${QMODEL_PATHS} \
--quant_model_bits ${Q_BITS} \
--config ${CONFIG} \
--bits ${BITS} \
-n ${N} \
--save ${SAVE} \
--datasets ${DATASETS} \
--method ${METHOD} \
--group_size ${GROUP_SIZE} \
# --zeroshot \
# --tasks ${TASKS} \
# --zeroshot_batch_size ${BATCH_SIZE}
# --clip_asym \
