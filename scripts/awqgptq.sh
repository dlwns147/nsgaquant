DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

Q_BITS="2 3 4"
Q_BITS_TEXT="234"

METHOD=awq
# METHOD=gptq

GROUP_SIZE=128

COMP_OBJ="bits"
COMP_OBJ_TEXT=bits


TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq openbookqa"
# BATCH_SIZE=32
BATCH_SIZE=64

N=1
DATASETS="wikitext2 c4"

GROUP_SIZE=128
# GROUP_SIZE=-1

SAVE=save/result/${TODAY}_${MODEL_NAME}_${COMP_OBJ}_${METHOD}_${BITS}

TARGET_COMP_OBJ=bits
# TARGET_BITS_LIST=(2 3 4)
BITS=1
N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} awqgptq.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--bits ${BITS} \
-n ${N} \
--save ${SAVE} \
--datasets ${DATASETS} \
--method ${METHOD} \
--zeroshot \
--tasks ${TASKS} \
--zeroshot_batch_size ${BATCH_SIZE} \
--group_size ${GROUP_SIZE} \
--clip_asym
