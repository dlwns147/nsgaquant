DEVICES=${1}

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf

# MODEL=facebook/opt-6.7b
# MODEL=facebook/opt-13b
# MODEL=facebook/opt-30b
# MODEL=facebook/opt-66b


BACKEND=BITBLAS
# BACKEND_SMALL=bitblas
# BACKEND=AUTO
# BACKEND_SMALL=auto
# BACKEND=CUDA

CONFIG=config/llama.json

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

TARGET_BIT=1e99
N_SAMPLE=128

LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_reverse_loss_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_reverse_ppl_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv

CUDA_VISIBLE_DEVICES=${DEVICES} python -m greedy_search_linear_reverse \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--large_model_bit ${LARGE_WBITS} \
--large_model_path ${LARGE_MODEL_PATH} \
--small_model_bit ${SMALL_WBITS} \
--small_model_path ${SMALL_MODEL_PATH} \
--backend ${BACKEND} \
--target_bit ${TARGET_BIT} \
--n_sample ${N_SAMPLE} \
--loss_csv_file ${LOSS_CSV_FILE} \
--ppl_csv_file ${PPL_CSV_FILE} \
--eval_ppl False \
--eval_zeroshot False \
--eval_ppl_iter True \
--config ${CONFIG}

# CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID