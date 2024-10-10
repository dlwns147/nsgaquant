DEVICES=${1}

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL=meta-llama/Llama-2-13b-hf
# MODEL=meta-llama/Llama-2-70b-hf

# MODEL=facebook/opt-6.7b
# MODEL=facebook/opt-13b
# MODEL=facebook/opt-30b
# MODEL=facebook/opt-66b

CONFIG=config/llama.json
N_SAMPLE=128

# BACKEND=BITBLAS
# BACKEND_SMALL=bitblas

# QUANT_METHOD=hqq
# LARGE_WBITS=4
# LARGE_GROUP_SIZE=128
# LARGE_AXIS=1
# LARGE_QSCALE=false
# LARGE_QZERO=false
# LARGE_MODEL_PATH=/SSD/${QUANT_METHOD}/Llama-2-7b-hf_${LARGE_WBITS}bit_${LARGE_GROUP_SIZE}gs_${LARGE_AXIS}axis_qscale_${LARGE_QSCALE}_qzero_${LARGE_QZERO}

# SMALL_WBITS=2
# SMALL_GROUP_SIZE=64
# # SMALL_GROUP_SIZE=128
# SMALL_AXIS=1
# SMALL_QSCALE=false
# SMALL_QZERO=false
# SMALL_MODEL_PATH=/SSD/hqq/Llama-2-7b-hf_${SMALL_WBITS}bit_${SMALL_GROUP_SIZE}gs_${SMALL_AXIS}axis_qscale_${SMALL_QSCALE}_qzero_${SMALL_QZERO}

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_loss_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_ppl_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv

QUANT_METHOD=owq
SMALL_WBITS=2.1
SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

LARGE_WBITS=4.1
LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth

LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${QUANT_METHOD}_loss_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${QUANT_METHOD}_ppl_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv

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

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${QUANT_METHOD}_loss_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${QUANT_METHOD}_ppl_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv

CUDA_VISIBLE_DEVICES=${DEVICES} python -m linear_sensitivity \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--quant_method ${QUANT_METHOD} \
--large_model_bit ${LARGE_WBITS} \
--large_model_path ${LARGE_MODEL_PATH} \
--small_model_bit ${SMALL_WBITS} \
--small_model_path ${SMALL_MODEL_PATH} \
--n_sample ${N_SAMPLE} \
--loss_csv_file ${LOSS_CSV_FILE} \
--ppl_csv_file ${PPL_CSV_FILE} \
--config ${CONFIG}
# CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID 
# --eval_ppl \
# --eval_zeroshot \
# CUDA_DEVICE_ORDER=PCI_BUS_ID 