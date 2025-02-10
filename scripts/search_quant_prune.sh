DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

METHOD="hqq layer_prune"
METHOD_TEXT="hqq_layer_prune"
# METHOD="hqq"
# METHOD_TEXT="hqq"

Q_BITS="2 3 4"
Q_BITS_TEXT="234"
# Q_BITS="2 4"
# Q_BITS_TEXT="24"
AXIS=1
GROUP_SIZE=128
QSCALE=false
QZERO=false

PASS_LINEAR_LIST="0.self_attn.v_proj 1.self_attn.v_proj 1.mlp.down_proj 31.mlp.down_proj" # Llama-2-7b
PASS_LAYER_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 31.mlp"

# PASS_LINEAR_LIST="0.self_attn.v_proj 0.mlp.down_proj 1.self_attn.v_proj 1.mlp.down_proj 2.self_attn.v_proj 3.self_attn.v_proj 3.mlp.down_proj 39.mlp.down_proj" # Llama-2-13b
# PASS_LAYER_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 3.self_attn 3.mlp 39.mlp"
QMODEL_PATHS_LIST=()
for B in ${Q_BITS}
do
    # QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_float16" )
done
# QMODEL_PATHS=( "/SSD/hqq/${MODEL_NAME}_2bit_64gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_3bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_4bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}")
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

# LOSS_FUNC=cross_entropy
LOSS_FUNC=jsd

PREDICTOR=rbf
# PREDICTOR=mlp

DATASET=wikitext2
# DATASET=c4

# N_SAMPLE=8
# N_SAMPLE=16
N_SAMPLE=32
# N_SAMPLE=64
# N_SAMPLE=128

OBJ=bits
SEC_OBJ_RANGE_SMALL=1.95
# SEC_OBJ_RANGE_SMALL=1.99
# SEC_OBJ_RANGE_SMALL=1.9
# SEC_OBJ_RANGE_SMALL=${Q_BITS:0:1}
SEC_OBJ_RANGE_LARGE=${Q_BITS:(-1)}

# OBJ=latency
# SEC_OBJ_RANGE_SMALL=1
# SEC_OBJ_RANGE_LARGE=1e3

# LAYER_PRUNE_RANGE_SMALL=0.001
# LAYER_PRUNE_RANGE_SMALL=0.7
# LAYER_PRUNE_RANGE_SMALL=0.9
LAYER_PRUNE_RANGE_SMALL=0.95
# # LAYER_PRUNE_RANGE_SMALL=1.0
LAYER_PRUNE_RANGE_LARGE=1.0

N_DOE=250
ITER=200

# N_DOE=300
# ITER=200

N_ITER=50
GA_POP_SIZE=200
METRIC=loss

MAX_VALUE=5
MUT_PROB=0.1
CROSSOVER_PROB=0.9

# LATENCY_TABLE=/NAS/JG/QAS4SD/llama2_7b_lpe_24bit_iter10000.json
# LATENCY_TABLE=/NAS/JG/QAS4SD/llama2_13b_lpe_24bit_iter10000.json

SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${PREDICTOR}
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${PREDICTOR}_outlier
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${DATASET}_${N_SAMPLE}sample_2_64
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${DATASET}_${N_SAMPLE}sample
# SAVE=save/search/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${GA_ALGORITHM}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_mut_${MUT_PROB}_layer_prune_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_linear_group

N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search_quant_prune.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--method ${METHOD} \
--quant_model_paths ${QMODEL_PATHS} \
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
--pass_layer_list ${PASS_LAYER_LIST} \
--max_value ${MAX_VALUE} \
--mut_prob ${MUT_PROB} \
--crossover_prob ${CROSSOVER_PROB} \
--pass_linear_list ${PASS_LINEAR_LIST} \
--loss_func ${LOSS_FUNC} \
--n_sample ${N_SAMPLE} \
--layer_prune_range ${LAYER_PRUNE_RANGE_SMALL} ${LAYER_PRUNE_RANGE_LARGE} \
--dataset ${DATASET}
# --base_outlier_bits ${OUTLIER_BITS} \
# --outlier_path ${OUTLIER_PATH} \
# --n_outlier ${N_OUTLIER}

# --base_outlier_bits ${OUTLIER_BITS} \
# --outlier_path ${OUTLIER_PATH} \
# --n_outlier ${N_OUTLIER}
# --latency_table_file ${LATENCY_TABLE}

# --base_outlier_bits ${OUTLIER_BITS} \
# --outlier_path ${OUTLIER_PATH} \
# --n_outlier ${N_OUTLIER} \
# --only_outlier_bits

# --use_linear_group
# --resume ${RESUME} \

# OUTLIER_BITS="2 3"
# N_OUTLIER=32
# OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth


# METHOD=gptq
# # BACKEND='BITBLAS'
# # BACKEND_SMALL='bitblas'
# BACKEND='AUTO'
# BACKEND_SMALL='auto'
# # BACKEND='QBITS'
# # BACKEND_SMALL='qbits'

# SMALL_WBITS=2
# SMALL_GROUP_SIZE=64
# SMALL_MODEL_PATH=/SSD/gptqmodel/${MODEL_NAME}_${SMALL_WBITS}bit_${SMALL_GROUP_SIZE}gs_${BACKEND_SMALL}
# LARGE_WBITS=4
# LARGE_GROUP_SIZE=128
# LARGE_MODEL_PATH=/SSD/gptqmodel/${MODEL_NAME}_${LARGE_WBITS}bit_${LARGE_GROUP_SIZE}gs_${BACKEND_SMALL}

# PASS_LINEAR_LIST="0.self_attn.q_proj 1.mlp.down_proj 3.self_attn.q_proj 4.self_attn.q_proj 4.mlp.up_proj 6.self_attn.q_proj 7.self_attn.q_proj 30.mlp.down_proj 31.mlp.down_proj"

# METHOD=owq
# SMALL_WBITS=2.01
# SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

# LARGE_WBITS=4.01
# LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_ppl_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv

# PASS_LINEAR_LIST="31.mlp.down_proj"
# MIN_SEC_OBJ=${SMALL_WBITS}
# MAX_SEC_OBJ=${LARGE_WBITS}


# METHOD=awq
# METHOD_TEXT=awq

# # METHOD="awq layer_prune"
# # METHOD_TEXT=awq_layer_prune

# Q_BITS="2 3 4"
# Q_BITS_TEXT=234
# GROUP_SIZE=128
# SCALE_BITS=2

# QMODEL_PATHS=()
# for B in ${Q_BITS}
# do
#     # QMODEL_PATHS+=( "/SSD/awq/${MODEL_NAME}_w${B}_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt" )
#     QMODEL_PATHS+=( "/SSD/awq/${MODEL_NAME}_w${B}_g${GROUP_SIZE}_fake_${SCALE_BITS}scale_asym.pt" )
# done
# PASS_LINEAR_LIST="31.mlp.down_proj"
# PASS_LINEAR_LIST="0.mlp.down_proj 39.mlp.up_proj 39.mlp.down_proj"

# PASS_LINEAR_LIST="30.mlp.up_proj 31.mlp.gate_proj 31.mlp.up_proj 31.mlp.down_proj 31.mlp.down_proj" # llama2 7b linear
# PASS_LINEAR_LIST="30.mlp.up_proj 31.mlp.up_proj 31.mlp.down_proj" # llama2 7b linear group
# PASS_LINEAR_LIST="21.mlp.up_proj 38.mlp.gate_proj 38.mlp.up_proj 38.mlp.down_proj 39.mlp.gate_proj 39.mlp.up_proj 39.mlp.down_proj" # llama2 13b linear
# PASS_LINEAR_LIST="38.mlp.up_proj 39.mlp.up_proj" # llama2 13b linear group
