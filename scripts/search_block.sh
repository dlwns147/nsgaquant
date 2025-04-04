DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

METHOD=layer_prune
METHOD_TEXT=layer_prune

# SEC_OBJ_RANGE_SMALL=0.4
SEC_OBJ_RANGE_SMALL=0.001
SEC_OBJ_RANGE_LARGE=1

# SEC_OBJ_RANGE_SMALL=0.001
# SEC_OBJ_RANGE_LARGE=1e9

LAYER_PRUNE_RANGE_SMALL=0.40
# LAYER_PRUNE_RANGE_SMALL=0.001
LAYER_PRUNE_RANGE_LARGE=1.0

LOSS_FUNC=cross_entropy
# LOSS_FUNC=jsd

PREDICTOR=mlp
# PREDICTOR=gp
# PREDICTOR=rbf

OBJ=sparsity
# OBJ=params
# OBJ=latency

# N_SAMPLE=64
N_SAMPLE=128

N_DOE=32
N_ITER=16
ITER=64
# N_DOE=64
# N_ITER=32
# ITER=128


# N_DOE=40
# N_ITER=20
# ITER=80
# N_DOE=80
# N_ITER=40
# ITER=160

GA_POP_SIZE=200
METRIC=loss

GA_ALGORITHM=nsga2
MAX_VALUE=10
MUT_PROB=0.1

Q_BITS=16

PASS_LAYER_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 31.mlp"
# PASS_LAYER_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 3.self_attn 3.mlp 39.mlp"
# PASS_LAYER_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 2.mlp 8.mlp 75.mlp 77.mlp 78.mlp 79.mlp"

LAYER_SENSITIVITY_FILE=csv/sensitivity/${MODEL_NAME}_layer_prune_loss_jsd.csv
PASS_LAYER_RATIO=0.1
# PASS_LAYER_RATIO=0.2
# PASS_LAYER_RATIO=0.3

LATENCY_TABLE=latency_table/${MODEL_NAME}_rtx6000ada.json

SAVE=save/search/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_n_iter_${N_ITER}_${GA_ALGORITHM}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_mut_${MUT_PROB}_mask_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${N_SAMPLE}sample_pass_ratio_${PASS_LAYER_RATIO}_block

N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search_block.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--method ${METHOD} \
--sec_obj ${OBJ} \
--predictor ${PREDICTOR} \
--save ${SAVE} \
--iterations ${ITER} \
--n_doe ${N_DOE} \
--save ${SAVE} \
--n_iter ${N_ITER} \
--metric ${METRIC} \
--ga_pop_size ${GA_POP_SIZE} \
--config ${CONFIG} \
--debug \
--sec_obj_range ${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE} \
--ga_algorithm ${GA_ALGORITHM} \
--max_value ${MAX_VALUE} \
--mut_prob ${MUT_PROB} \
--layer_prune_range ${LAYER_PRUNE_RANGE_SMALL} ${LAYER_PRUNE_RANGE_LARGE} \
--loss_func ${LOSS_FUNC} \
--latency_table_file ${LATENCY_TABLE} \
--n_sample ${N_SAMPLE} \
--pass_layer_list ${PASS_LAYER_LIST}
# --layer_sensitivity_file ${LAYER_SENSITIVITY_FILE} \
# --pass_layer_ratio ${PASS_LAYER_RATIO}
# --resume ${RESUME}
