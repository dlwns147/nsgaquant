DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

METHOD=layer_prune
METHOD_TEXT=layer_prune

COMP_OBJ="params latency"
COMP_OBJ_TEXT=params_latency

COMP_OBJ_MIN="0.45 0.45" 
COMP_OBJ_MIN_TEXT=0.45_0.45

COMP_OBJ_MAX="1 1"
COMP_OBJ_MAX_TEXT=1_1

# LAYER_PRUNE_RANGE_SMALL=0.45
# # LAYER_PRUNE_RANGE_SMALL=0.001
# LAYER_PRUNE_RANGE_LARGE=1.0

# LOSS_FUNC=cross_entropy
LOSS_FUNC=jsd

PREDICTOR=mlp
# PREDICTOR=gp
# PREDICTOR=rbf

DATASET=wikitext2
# DATASET=c4

# N_SAMPLE=16
N_SAMPLE=32
# N_SAMPLE=64
# N_SAMPLE=128

# N_DOE=64
# N_ITER=32
# # ITER=128
# ITER=192

N_DOE=80
N_ITER=40
# ITER=160
ITER=240

# N_DOE=160
# N_ITER=80
# ITER=320

# GA_POP_SIZE=64
# GA_POP_SIZE=96
GA_POP_SIZE=200
METRIC=loss

MAX_VALUE=10
# MAX_VALUE=5
MUT_PROB=0.1
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

SAVE=save/search/prune/${TODAY}_${MODEL_NAME}_mo_${COMP_OBJ_TEXT}_${METRIC}_iter_${ITER}_n_iter_${N_ITER}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_mask_${N_SAMPLE}sample_pop_${GA_POP_SIZE}_${DATASET}_${ENV}
# ${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}

N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search_layer_multi_obj.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--dataset ${DATASET} \
--method ${METHOD} \
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${COMP_OBJ_MIN} \
--comp_obj_max ${COMP_OBJ_MAX} \
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
--max_value ${MAX_VALUE} \
--mut_prob ${MUT_PROB} \
--crossover_prob ${CROSSOVER_PROB} \
--loss_func ${LOSS_FUNC} \
--latency_table_file ${LATENCY_TABLE} \
--n_sample ${N_SAMPLE} \
--pass_layer_list ${PASS_LAYER_LIST}
# --layer_sensitivity_file ${LAYER_SENSITIVITY_FILE} \
# --pass_layer_ratio ${PASS_LAYER_RATIO}
# --resume ${RESUME}
# --layer_prune_range ${LAYER_PRUNE_RANGE_SMALL} ${LAYER_PRUNE_RANGE_LARGE} \
