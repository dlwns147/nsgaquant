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
METHOD=layer_prune
ITERATION=10
BATCH_SIZE=64
DATA_FILE=/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_layer_prune_loss_0.5_1_jsd.json
# LATENCY_FILE=data/${MODEL_NAME}_latency_gen_${BATCH_SIZE}bs_${ITERATION}iter_arch_lat_sparsity_gs10_7.json
LATENCY_FILE=data/${MODEL_NAME}_latency_table_gen_${BATCH_SIZE}bs_${ITERATION}iter_arch_lat_gs10_6.json

# METHOD=hqq
# LARGE_WBITS=4
# LARGE_GROUP_SIZE=128
# LARGE_AXIS=1
# LARGE_QSCALE=false
# LARGE_QZERO=false
# LARGE_MODEL_PATH=/SSD/hqq/${MODEL_NAME}_${LARGE_WBITS}bit_${LARGE_GROUP_SIZE}gs_${LARGE_AXIS}axis_qscale_${LARGE_QSCALE}_qzero_${LARGE_QZERO}

# SMALL_WBITS=2
# # SMALL_GROUP_SIZE=64
# SMALL_GROUP_SIZE=128
# SMALL_AXIS=1
# SMALL_QSCALE=false
# SMALL_QZERO=false
# SMALL_MODEL_PATH=/SSD/hqq/${MODEL_NAME}_${SMALL_WBITS}bit_${SMALL_GROUP_SIZE}gs_${SMALL_AXIS}axis_qscale_${SMALL_QSCALE}_qzero_${SMALL_QZERO}

# METHOD=gptq
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

# METHOD=owq
# SMALL_WBITS=2.1
# SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

# LARGE_WBITS=4.1
# LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth

# LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${METHOD}_loss_desc_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${METHOD}_ppl_desc_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# LOSS_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${METHOD}_loss_asc_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# PPL_CSV_FILE=csv/greedy_search/${MODEL_NAME}_${METHOD}_ppl_asc_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv

CUDA_VISIBLE_DEVICES=${DEVICES} python -m latency \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--config ${CONFIG} \
--iteration ${ITERATION} \
--batch_size ${BATCH_SIZE} \
--latency_file ${LATENCY_FILE} \
--method ${METHOD} \
--data ${DATA_FILE}

# --quant_method ${METHOD} \
# --large_model_bit ${LARGE_WBITS} \
# --large_model_path ${LARGE_MODEL_PATH} \
# --small_model_bit ${SMALL_WBITS} \
# --small_model_path ${SMALL_MODEL_PATH} \
# --eval_ppl \
# --eval_zeroshot \

# --backend ${BACKEND} \
# CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
