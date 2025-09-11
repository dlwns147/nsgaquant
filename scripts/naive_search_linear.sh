DEVICES=${1}

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf
DTYPE=float16

# MODEL=facebook/opt-6.7b
# MODEL=facebook/opt-13b
# MODEL=facebook/opt-30b
# MODEL=facebook/opt-66b

CONFIG=config/llama.json
N_SAMPLE=128

GROUP_SIZE=128
AXIS=1

QUANT_METHOD=hqq

LARGE_WBITS=2
LARGE_MODEL_PATH=/SSD/hqq/${MODEL_NAME}_${LARGE_WBITS}bit_${GROUP_SIZE}gs_${AXIS}axis_${DTYPE}

SMALL_WBITS=2
SMALL_MODEL_PATH=/SSD/hqq/${MODEL_NAME}_${SMALL_WBITS}bit_${GROUP_SIZE}gs_${AXIS}axis_${DTYPE}

# # LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_loss_desc_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
# # PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_desc_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
# LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_loss_asc_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
# PPL_CSV_FILE=csv/naive_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv

PPL_CSV_FILE=csv/naive_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_${LARGE_WBITS}_${SMALL_WBITS}_bits_axis_${AXIS}.csv
LINEAR_SENSITIVITY_FILE=/NAS/SJ/nsgaquant/csv/sensitivity/${MODEL_NAME}_hqq_loss_24_1axis_128gs_jsd_wikitext2_128sample.csv

# TARGET_BIT=-1
TARGET_BIT=3.5

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
--linear_sensitivity ${LINEAR_SENSITIVITY_FILE}

# --eval_ppl \
# --eval_zeroshot \

# --backend ${BACKEND} \
# CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
