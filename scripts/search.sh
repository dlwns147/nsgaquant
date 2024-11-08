DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=config/llama.json

# METHOD="hqq"
# METHOD_TEXT="hqq"

# Q_BITS="2 3 4"
# Q_BITS_TEXT="234"
# AXIS=1
# GROUP_SIZE=128
# QSCALE=false
# QZERO=false
# PASS_LIST="0.self_attn.v_proj 1.self_attn.v_proj 1.mlp.down_proj 31.mlp.down_proj"

# QMODEL_PATHS=()
# # echo ${QMODEL_PATHS}
# for B in ${Q_BITS}
# do
#     # echo "/SSD/hqq/Llama-2-7b-hf_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}"
#     QMODEL_PATHS+=( "/SSD/hqq/Llama-2-7b-hf_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
#     # echo ${QMODEL_PATHS}
# done

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

# PASS_LIST="0.self_attn.q_proj 1.mlp.down_proj 3.self_attn.q_proj 4.self_attn.q_proj 4.mlp.up_proj 6.self_attn.q_proj 7.self_attn.q_proj 30.mlp.down_proj 31.mlp.down_proj"

# METHOD=owq
# SMALL_WBITS=2.01
# SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

# LARGE_WBITS=4.01
# LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_ppl_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv

# PASS_LIST="31.mlp.down_proj"
# MIN_SEC_OBJ=${SMALL_WBITS}
# MAX_SEC_OBJ=${LARGE_WBITS}

# METHOD=awq
# METHOD_TEXT="awq"
# PASS_LIST="31.mlp.down_proj"

METHOD=awq
METHOD_TEXT=awq
Q_BITS="2 3 4"
Q_BITS_TEXT=234
GROUP_SIZE=128
SCALE_BITS=4

QMODEL_PATHS=()
# echo ${QMODEL_PATHS}
for B in ${Q_BITS}
do
    # echo "/SSD/hqq/Llama-2-7b-hf_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}"
    QMODEL_PATHS+=( "/SSD/awq/${MODEL_NAME}_w${B}_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt" )
    # echo ${QMODEL_PATHS}
done

PASS_LIST="31.mlp.down_proj"

SEC_OBJ_RANGE_SMALL=${Q_BITS:0:1} 
SEC_OBJ_RANGE_LARGE=${Q_BITS:(-1)}
SEC_OBJ_RANGE="${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE}"

LAYER_PRUNE_RANGE_SMALL=1.0
LAYER_PRUNE_RANGE_LARGE=1.0
LAYER_PRUNE_RANGE="${LAYER_PRUNE_RANGE_SMALL} ${LAYER_PRUNE_RANGE_LARGE}"

PREDICTOR=mlp
OBJ=bits
DEBUG=True

N_DOE=250
N_ITER=50
ITER=300
GA_POP_SIZE=200
METRIC=loss

GA_ALGORITHM='nsga2'
# GA_ALGORITHM='ga'
MAX_VALUE=5
MUT_PROB=0.05

SAVE=save/search/${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${GA_ALGORITHM}_${Q_BITS_TEXT}_${SCALE_BITS}scale_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_mut_${MUT_PROB}_layer_prune_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${TODAY}
# SAVE=save/search/${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${GA_ALGORITHM}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_mut_${MUT_PROB}_layer_prune_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${TODAY}

# RESUME=/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_awq_bits_loss_234_128gs_4scale_2_4_lp_1.0_1.0.json
N_PROC=2

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search.py \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--method ${METHOD} \
--quant_model_paths "${QMODEL_PATHS[@]}" \
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
--debug ${DEBUG} \
--sec_obj_range ${SEC_OBJ_RANGE} \
--ga_algorithm ${GA_ALGORITHM} \
--max_value ${MAX_VALUE} \
--mut_prob ${MUT_PROB} \
--pass_linear_list ${PASS_LIST} \
--layer_prune_range ${LAYER_PRUNE_RANGE}
# --resume ${RESUME}


# --large_model_path ${LARGE_MODEL_PATH} \
# --large_model_bits ${LARGE_WBITS} \
# --small_model_path ${SMALL_MODEL_PATH} \
# --small_model_bits ${SMALL_WBITS} \
# --sec_n_doe ${SEC_N_DOE} \
# --sec_n_iter ${SEC_N_ITER} \
# --sec_iter ${SEC_ITER} \
# --sec_metric ${SEC_METRIC} \
# --sec_ga_pop_size ${SEC_GA_POP_SIZE} \
# --resume ${RESUME}


# MIN_SEC_OBJ=2.19
# MAX_SEC_OBJ=2.21

# MIN_SEC_OBJ=2.39
# MAX_SEC_OBJ=2.41

# MIN_SEC_OBJ=2.59
# MAX_SEC_OBJ=2.61

# MIN_SEC_OBJ=2.79
# MAX_SEC_OBJ=2.81

# MIN_SEC_OBJ=2.99
# MAX_SEC_OBJ=3.01

# MIN_SEC_OBJ=3.19
# MAX_SEC_OBJ=3.21

# MIN_SEC_OBJ=3.39
# MAX_SEC_OBJ=3.41

# MIN_SEC_OBJ=3.59
# MAX_SEC_OBJ=3.61

# MIN_SEC_OBJ=3.79
# MAX_SEC_OBJ=3.81

# MIN_SEC_OBJ=2.495
# MAX_SEC_OBJ=2.505

# MIN_SEC_OBJ=2.745
# MAX_SEC_OBJ=2.755

# MIN_SEC_OBJ=2.995
# MAX_SEC_OBJ=3.005

# MIN_SEC_OBJ=3.495
# MAX_SEC_OBJ=3.505

# PREDICTOR_DATA=data/Llama-2-7b-hf_loss_1000_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json
# PREDICTOR_DATA=data/Llama-2-7b-hf_loss_300_range_2.9_3.1_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json
# PREDICTOR_DATA=data/Llama-2-7b-hf_loss_300_range_2_4_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json

# SEC_N_DOE=50
# SEC_N_ITER=20
# SEC_ITER=20
# SEC_METRIC=ppl
# SEC_GA_POP_SIZE=50

# SEC_N_DOE=50
# SEC_N_ITER=20
# SEC_ITER=30
# SEC_METRIC=ppl
# SEC_GA_POP_SIZE=50
