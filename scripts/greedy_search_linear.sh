DEVICES=${1}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf

# MODEL=facebook/opt-6.7b
# MODEL=facebook/opt-13b
# MODEL=facebook/opt-30b
# MODEL=facebook/opt-66b

CONFIG=config/llama.json
N_SAMPLE=128

# LOSS_FUNC=cross_entropy
LOSS_FUNC=jsd

METHOD="hqq"
METHOD_TEXT="hqq"

Q_BITS="3 4"
Q_BITS_TEXT=34
AXIS=1
GROUP_SIZE=128

QMODEL_PATHS=()
for B in ${Q_BITS}
do
    QMODEL_PATHS+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_false_qzero_false" )
done
# QMODEL_PATHS=( "/SSD/hqq/${MODEL_NAME}_2bit_64gs_${AXIS}axis_qscale_false_qzero_false" "/SSD/hqq/${MODEL_NAME}_3bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_false_qzero_false" )


LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${METHOD_TEXT}_${Q_BITS_TEXT}bits_loss_desc_1axis_64_128gs_${LOSS_FUNC}.csv
PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${METHOD_TEXT}_${Q_BITS_TEXT}bits_ppl_desc_1axis_64_128gs_${LOSS_FUNC}.csv
# LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_loss_asc_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv
# PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${QUANT_METHOD}_ppl_asc_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.csv

TARGET_BIT=-1
# TARGET_BIT=16

N_PROC=2

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} -m greedy_search_linear \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--method ${METHOD} \
--quant_model_paths "${QMODEL_PATHS[@]}" \
--quant_model_bits ${Q_BITS} \
--target_bit ${TARGET_BIT} \
--n_sample ${N_SAMPLE} \
--loss_csv_file ${LOSS_CSV_FILE} \
--ppl_csv_file ${PPL_CSV_FILE} \
--config ${CONFIG} \
--loss_func ${LOSS_FUNC} \
--eval_ppl_iter \
--descending
# --eval_ppl \
# --eval_zeroshot \

# --backend ${BACKEND} \
# CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID


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
