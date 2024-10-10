DEVICES=${1}
TODAY=`date +%y%m%d%H%M`

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=config/llama.json

QUANT_METHOD=hqq
LARGE_WBITS=4
LARGE_GROUP_SIZE=128
LARGE_AXIS=1
LARGE_QSCALE=false
LARGE_QZERO=false
LARGE_MODEL_PATH=/SSD/hqq/Llama-2-7b-hf_${LARGE_WBITS}bit_${LARGE_GROUP_SIZE}gs_${LARGE_AXIS}axis_qscale_${LARGE_QSCALE}_qzero_${LARGE_QZERO}

SMALL_WBITS=2
SMALL_GROUP_SIZE=128
# SMALL_GROUP_SIZE=64
SMALL_AXIS=1
SMALL_QSCALE=false
SMALL_QZERO=false
SMALL_MODEL_PATH=/SSD/hqq/Llama-2-7b-hf_${SMALL_WBITS}bit_${SMALL_GROUP_SIZE}gs_${SMALL_AXIS}axis_qscale_${SMALL_QSCALE}_qzero_${SMALL_QZERO}

# PASS_LIST="1.self_attn.v_proj 1.mlp.down_proj 31.mlp.down_proj"

PASS_LIST="0.self_attn.v_proj 1.self_attn.v_proj 1.mlp.down_proj 31.mlp.down_proj"

# QUANT_METHOD=gptq
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

MIN_SEC_OBJ=${SMALL_WBITS}
MAX_SEC_OBJ=${LARGE_WBITS}

# QUANT_METHOD=owq
# SMALL_WBITS=2.01
# SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

# LARGE_WBITS=4.01
# LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${QUANT_METHOD}_loss_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${QUANT_METHOD}_ppl_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv

# PASS_LIST="31.mlp.down_proj"
# MIN_SEC_OBJ=${SMALL_WBITS}
# MAX_SEC_OBJ=${LARGE_WBITS}

PREDICTOR=mlp
OBJ=bits
DEBUG=True

N_DOE=250
N_ITER=50
ITER=300
GA_POP_SIZE=100
METRIC=loss


GA_ALGORITHM='nsga2'
# GA_ALGORITHM='ga'
NAN_VALUE=10

SAVE=save/search/${MODEL_NAME}_${OBJ}_${METRIC}_${QUANT_METHOD}_iter_${ITER}_${GA_ALGORITHM}_${MIN_SEC_OBJ}_${MAX_SEC_OBJ}_mut_prob_0.2_${TODAY}
# RESUME=/NAS/SJ/nsgaquant/save/search/Llama-2-7b-hf_bits_loss_iter_300_nsga2_2_4_2410051059/iter_100.stats

CUDA_VISIBLE_DEVICES=${DEVICES} python search.py \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--large_model_path ${LARGE_MODEL_PATH} \
--large_model_bits ${LARGE_WBITS} \
--small_model_path ${SMALL_MODEL_PATH} \
--small_model_bits ${SMALL_WBITS} \
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
--sec_obj_range ${MIN_SEC_OBJ} ${MAX_SEC_OBJ} \
--ga_algorithm ${GA_ALGORITHM} \
--quant_method ${QUANT_METHOD} \
--pass_linear_list ${PASS_LIST} \
--nan_value ${NAN_VALUE}


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
# SEC_GA