DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

METHOD=layer_prune
METHOD_TEXT=layer_prune

SEC_OBJ_RANGE_SMALL=0.4
# SEC_OBJ_RANGE_SMALL=0.01
# SEC_OBJ_RANGE_SMALL=0.45
SEC_OBJ_RANGE_LARGE=1

# SEC_OBJ_RANGE_SMALL=0.4
# SEC_OBJ_RANGE_LARGE=1.0

LAYER_PRUNE_RANGE_SMALL=0.4
# LAYER_PRUNE_RANGE_SMALL=0.001
LAYER_PRUNE_RANGE_LARGE=1.0

# LOSS_FUNC=cross_entropy
LOSS_FUNC=jsd

PREDICTOR=mlp
# PREDICTOR=gp
# PREDICTOR=rbf

OBJ=sparsity
# OBJ=params
# OBJ=latency

DATASET=wikitext2
# DATASET=c4

# N_SAMPLE=16
# N_SAMPLE=32
# N_SAMPLE=64
N_SAMPLE=128

# N_ITER=10
# N_ITER=20

# N_DOE=64
# N_ITER=32
# ITER=128
# ITER=192

# GA_POP_SIZE=32

N_DOE=80
N_ITER=40
ITER=160
# ITER=240

# GA_POP_SIZE=40

# N_DOE=160
# N_ITER=80
# ITER=320

# GA_POP_SIZE=64
# GA_POP_SIZE=96
# GA_POP_SIZE=50
# GA_POP_SIZE=100
GA_POP_SIZE=200

# SUBSET_POP_SIZE=50
SUBSET_POP_SIZE=100


METRIC=loss

# MAX_VALUE=10
MAX_VALUE=5
MUT_PROB=0.1
# MUT_PROB=0.2
CROSSOVER_PROB=0.9
# CROSSOVER_PROB=1.0


Q_BITS=16

# PASS_LAYER_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 31.mlp"
PASS_LAYER_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 3.self_attn 3.mlp 39.mlp"
# PASS_LAYER_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 2.mlp 8.mlp 75.mlp 77.mlp 78.mlp 79.mlp"

# LAYER_SENSITIVITY_FILE=csv/sensitivity/${MODEL_NAME}_layer_prune_loss_jsd.csv
# PASS_LAYER_RATIO=0.1

ENV=rtx6000ada
# ENV=a100
LATENCY_TABLE=latency_table/${MODEL_NAME}_${ENV}_token.json

SAVE=save/search/prune/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_n_iter_${N_ITER}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_mask_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${N_SAMPLE}sample_pop_${GA_POP_SIZE}_${SUBSET_POP_SIZE}_${DATASET}_${PREDICTOR}_${ENV}

N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search_layer.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--dataset ${DATASET} \
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
--subset_pop_size ${SUBSET_POP_SIZE} \
--config ${CONFIG} \
--debug \
--sec_obj_range ${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE} \
--max_value ${MAX_VALUE} \
--mut_prob ${MUT_PROB} \
--crossover_prob ${CROSSOVER_PROB} \
--layer_prune_range ${LAYER_PRUNE_RANGE_SMALL} ${LAYER_PRUNE_RANGE_LARGE} \
--loss_func ${LOSS_FUNC} \
--latency_table_file ${LATENCY_TABLE} \
--n_sample ${N_SAMPLE} \
--pass_layer_list ${PASS_LAYER_LIST}
# --layer_sensitivity_file ${LAYER_SENSITIVITY_FILE} \
# --pass_layer_ratio ${PASS_LAYER_RATIO}
# --resume ${RESUME}
