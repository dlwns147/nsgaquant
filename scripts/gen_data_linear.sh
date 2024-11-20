DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=config/llama.json

N_DATA=250
# N_DATA=500
# N_DATA=1000
N_SAMPLE=128

# METHOD="hqq layer_prune"
# METHOD_TEXT=hqq_layer_prune

# Q_BITS="2 4"
# Q_BITS_TEXT="2_4"
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

# SEC_OBJ=bits
# SEC_OBJ_RANGE_SMALL=${Q_BITS:0:1} 
# SEC_OBJ_RANGE_LARGE=${Q_BITS:(-1)}
# SEC_OBJ_RANGE="${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE}"
# LAYER_PRUNE_RANGE_SMALL=0.95
# LAYER_PRUNE_RANGE_LARGE=1.0
# LAYER_PRUNE_RANGE="${LAYER_PRUNE_RANGE_SMALL} ${LAYER_PRUNE_RANGE_LARGE}"

# LOSS_FILE=data/${MODEL_NAME}_${METHOD_TEXT}_loss_${N_DATA}_range_bits_${Q_BITS_TEXT}_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_axis_${AXIS}_scale_${QSCALE}_qz_${QZERO}_lp_range_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}.json
# PPL_FILE=data/${MODEL_NAME}_${METHOD_TEXT}_ppl_${N_DATA}_range_bits_${Q_BITS_TEXT}_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_axis_${AXIS}_scale_${QSCALE}_qz_${QZERO}_lp_range_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}.json

# LOSS_FILE=data/${MODEL_NAME}_${METHOD}_loss_${N_DATA}_range_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.json
# PPL_FILE=data/${MODEL_NAME}_${METHOD}_ppl_${N_DATA}_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.json

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

# LOSS_FILE=data/${MODEL_NAME}_${METHOD}_loss_${N_DATA}_range_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv
# PPL_FILE=data/${MODEL_NAME}_${METHOD}_ppl_${N_DATA}_range_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv

# PASS_LIST="0.self_attn.q_proj 1.mlp.down_proj 3.self_attn.q_proj 4.self_attn.q_proj 4.mlp.up_proj 6.self_attn.q_proj 7.self_attn.q_proj 30.mlp.down_proj 31.mlp.down_proj 1.self_attn.v_proj 2.self_attn.q_proj"

# SEC_OBJ_RANGE_SMALL=2
# SEC_OBJ_RANGE_LARGE=4
# SEC_OBJ_RANGE="${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE}"

# METHOD=owq
# SMALL_WBITS=2.01
# SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

# LARGE_WBITS=4.01
# LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth

# LOSS_FILE=data/${MODEL_NAME}_${METHOD}_loss_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# PPL_FILE=data/${MODEL_NAME}_${METHOD}_ppl_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv

# PASS_LIST="31.mlp.down_proj"
# SEC_OBJ_RANGE_SMALL=${SMALL_WBITS}
# SEC_OBJ_RANGE_LARGE=${LARGE_WBITS}
# SEC_OBJ_RANGE="${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE}"

METHOD=awq
Q_BITS="2 3 4"
Q_BITS_TEXT="234"
SCALE_BITS=3
GROUP_SIZE=128

QMODEL_PATHS=()
echo ${QMODEL_PATHS}
for B in ${Q_BITS}
do
    # echo "/SSD/hqq/Llama-2-7b-hf_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}"
    QMODEL_PATHS+=( "/SSD/awq/${MODEL_NAME}_w${B}_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt" )
    # echo ${QMODEL_PATHS}
done

# QMODEL_PATHS=( "/SSD/awq/${MODEL_NAME}_w2_g64_fake_4bit_128gs_awq.pt" "/SSD/awq/${MODEL_NAME}_w4_g128_fake_4bit_awq.pt" )
# QMODEL_PATHS=( "/SSD/awq/${MODEL_NAME}_w2_g128_fake_2bit_awq.pt" "/SSD/awq/${MODEL_NAME}_w4_g128_fake_4bit_awq.pt" )

PASS_LIST="0.self_attn.q_proj 1.self_attn.q_proj 31.mlp.down_proj"

SEC_OBJ=bits
SEC_OBJ_RANGE_SMALL=${Q_BITS:0:1} 
SEC_OBJ_RANGE_LARGE=${Q_BITS:(-1)}
SEC_OBJ_RANGE="${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE}"
LAYER_PRUNE_RANGE_SMALL=1.0
LAYER_PRUNE_RANGE_LARGE=1.0
LAYER_PRUNE_RANGE="${LAYER_PRUNE_RANGE_SMALL} ${LAYER_PRUNE_RANGE_LARGE}"

LOSS_FILE=data/${MODEL_NAME}_${METHOD}_${SEC_OBJ}_loss_${Q_BITS_TEXT}_group_${GROUP_SIZE}gs_${SCALE_BITS}scale_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}.json
PPL_FILE=data/${MODEL_NAME}_${METHOD}_${SEC_OBJ}_ppl_${Q_BITS_TEXT}_group_${GROUP_SIZE}gs_${SCALE_BITS}scale_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}.json

MAX_VALUE=10
N_PROC=2

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} gen_data_linear.py \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--quant_model_paths "${QMODEL_PATHS[@]}" \
--quant_model_bits ${Q_BITS} \
--loss_json_file ${LOSS_FILE} \
--ppl_json_file ${PPL_FILE} \
--n_data ${N_DATA} \
--n_sample ${N_SAMPLE} \
--sec_obj ${SEC_OBJ} \
--sec_obj_range ${SEC_OBJ_RANGE} \
--method ${METHOD} \
--max_value ${MAX_VALUE} \
--layer_prune_range ${LAYER_PRUNE_RANGE} \
--use_linear_group
# --pass_linear_list ${PASS_LIST} \
# --large_model_path ${LARGE_MODEL_PATH} \
# --large_model_bits ${LARGE_WBITS} \
# --small_model_path ${SMALL_MODEL_PATH} \
# --small_model_bits ${SMALL_WBITS} \