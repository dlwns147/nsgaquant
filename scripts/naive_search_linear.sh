DEVICES=${1}

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf

# MODEL=facebook/opt-6.7b
# MODEL=facebook/opt-13b
# MODEL=facebook/opt-30b
# MODEL=facebook/opt-66b

CONFIG=config/llama.json
N_SAMPLE=128

QUANT_METHOD=hqq
LARGE_WBITS=4
LARGE_GROUP_SIZE=128
LARGE_AXIS=1
LARGE_QSCALE=false
LARGE_QZERO=false
LARGE_MODEL_PATH=/SSD/hqq/Llama-2-7b-hf_${LARGE_WBITS}bit_${LARGE_GROUP_SIZE}gs_${LARGE_AXIS}axis_qscale_${LARGE_QSCALE}_qzero_${LARGE_QZERO}

SMALL_WBITS=2
# SMALL_GROUP_SIZE=64
SMALL_GROUP_SIZE=128
SMALL_AXIS=1
SMALL_QSCALE=false
SMALL_QZERO=false
SMALL_MODEL_PATH=/SSD/hqq/Llama-2-7b-hf_${SMALL_WBITS}bit_${SMALL_GROUP_SIZE}gs_${SMALL_AXIS}axis_qscale_${SMALL_QSCALE}_qzero_${SMALL_QZERO}

# # LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_loss_desc_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
# # PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_desc_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
# LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_loss_asc_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
PPL_CSV_FILE=csv/naive_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
LINEAR_SENSITIVITY=csv/sensitivity/Llama-2-7b-hf_loss_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_128_sqs_false_sqz_false.csv
# QUANT_METHOD=gptq
# # BACKEND='BITBLAS'
# # BACKEND_SMALL='bitblas'
# BACKEND='AUTO'
# BACKEND_SMALL='auto'
# # BACKEND='QBITS'
# # BACKEND_SMALL='qbits'

# MODEL_PATH=meta-llama
# MODEL_NAME=Llama-2-7b-hf
# SMALL_WBITS=2
# SMALL_GROUP_SIZE=64
# SMALL_MODEL_PATH=/SSD/gptqmodel/${MODEL_NAME}_${SMALL_WBITS}bit_${SMALL_GROUP_SIZE}gs_${BACKEND_SMALL}
# LARGE_WBITS=4
# LARGE_GROUP_SIZE=128
# LARGE_MODEL_PATH=/SSD/gptqmodel/${MODEL_NAME}_${LARGE_WBITS}bit_${LARGE_GROUP_SIZE}gs_${BACKEND_SMALL}

# # LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_loss_desc_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv
# # PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_desc_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv
# LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_loss_asc_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv
# PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_asc_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv

# QUANT_METHOD=owq
# SMALL_WBITS=2.1
# SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

# LARGE_WBITS=4.1
# LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth

# LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_loss_desc_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_desc_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_loss_asc_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_asc_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv

# PASS_LIST="31.mlp.down_proj"


# TARGET_BIT=-1
# TARGET_BIT=16

CUDA_VISIBLE_DEVICES=${DEVICES} python -m naive_search_linear \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--large_model_bit ${LARGE_WBITS} \
--large_model_path ${LARGE_MODEL_PATH} \
--small_model_bit ${SMALL_WBITS} \
--small_model_path ${SMALL_MODEL_PATH} \
--n_sample ${N_SAMPLE} \
--ppl_csv_file ${PPL_CSV_FILE} \
--quant_method ${QUANT_METHOD} \
--config ${CONFIG} \
--linear_sensitivity ${LINEAR_SENSITIVITY}

# --eval_ppl \
# --eval_zeroshot \

# --backend ${BACKEND} \
# CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
