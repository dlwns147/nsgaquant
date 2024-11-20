DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=config/llama.json

METHOD=layer_prune
METHOD_TEXT=layer_prune

SEC_OBJ_RANGE_SMALL=0.5
# SEC_OBJ_RANGE_SMALL=0.001
SEC_OBJ_RANGE_LARGE=1.

LAYER_PRUNE_RANGE_SMALL=0.01
# LAYER_PRUNE_RANGE_SMALL=0.5
LAYER_PRUNE_RANGE_LARGE=1.0

# LOSS_FUNC=cross_entropy
LOSS_FUNC=jsd

PREDICTOR=mlp
# OBJ=sparsity
OBJ=params

Q_BITS=16

N_DOE=64
N_ITER=32
ITER=128
GA_POP_SIZE=200
METRIC=loss

GA_ALGORITHM=nsga2
MAX_VALUE=10
MUT_PROB=0.1

Q_BITS=16

PASS_LAYER_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 31.mlp"

SAVE=save/search/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_n_iter_${N_ITER}_${GA_ALGORITHM}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_mut_${MUT_PROB}_layer_prune_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}

N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--method ${METHOD} \
--quant_model_bits ${Q_BITS} \
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
--pass_layer_list ${PASS_LAYER_LIST} \
--layer_prune_range ${LAYER_PRUNE_RANGE_SMALL} ${LAYER_PRUNE_RANGE_LARGE} \
--loss_func ${LOSS_FUNC}
# --resume ${RESUME}
