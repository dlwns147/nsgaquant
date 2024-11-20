DEVICES=${1}

MODEL_PATH=meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
# MODEL=meta-llama/Llama-2-13b-hf
# MODEL=meta-llama/Llama-2-70b-hf

# MODEL=facebook/opt-6.7b
# MODEL=facebook/opt-13b
# MODEL=facebook/opt-30b
# MODEL=facebook/opt-66b

CONFIG=config/llama.json
N_SAMPLE=128

METHOD="hqq"
METHOD_TEXT="hqq"

Q_BITS="2 4"
Q_BITS_TEXT="24"
AXIS=1
GROUP_SIZE=128
QSCALE=false
QZERO=false

QMODEL_PATHS=()
for B in ${Q_BITS}
do
    QMODEL_PATHS+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_false_qzero_false" )
done
LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_loss_${Q_BITS_TEXT}_${AXIS}axis_${GROUP_SIZE}gs_${QSCALE}_qs_${QZERO}_qz.csv
PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_ppl_${Q_BITS_TEXT}_${AXIS}axis_${GROUP_SIZE}gs_${QSCALE}_qs_${QZERO}_qz.csv

# METHOD=owq
# SMALL_WBITS=2.1
# SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

# LARGE_WBITS=4.1
# LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_ppl_lb_${LARGE_WBITS}_sb_${SMALL_WBITS}.csv

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

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_ppl_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_sb_${SMALL_WBITS}_sgs${SMALL_GROUP_SIZE}.csv

# METHOD=awq
# Q_BITS="2 4"
# Q_BITS_TEXT="24"
# SCALE_BITS=3
# GROUP_SIZE=128
# CONFIG=config/llama.json
# # CONFIG=config/llama_awq.json


# # LOSS_FUNC=cross_entropy
# LOSS_FUNC=jsd
# QMODEL_PATHS=()

# for B in ${Q_BITS}
# do
#     # echo "/SSD/hqq/Llama-2-7b-hf_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}"
#     QMODEL_PATHS+=( "/SSD/awq/${MODEL_NAME}_w${B}_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt" )
#     # echo ${QMODEL_PATHS}
# done

# QMODEL_PATHS=("/SSD/awq/${MODEL_NAME}_w2_g64_fake_${SCALE_BITS}bit_128gs_awq.pt" "/SSD/awq/${MODEL_NAME}_w4_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt")

# QMODEL_PATHS=( "/SSD/awq/${MODEL_NAME}_w2_g64_fake_4bit_128gs_awq.pt" "/SSD/awq/${MODEL_NAME}_w4_g128_fake_4bit_awq.pt" )
# QMODEL_PATHS=( "/SSD/awq/${MODEL_NAME}_w2_g128_fake_2bit_awq.pt" "/SSD/awq/${MODEL_NAME}_w4_g128_fake_4bit_awq.pt" )

LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_${Q_BITS_TEXT}_${GROUP_SIZE}gs_${SCALE_BITS}scale_${LOSS_FUNC}.csv
PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_ppl_${Q_BITS_TEXT}_${GROUP_SIZE}gs_${SCALE_BITS}scale_${LOSS_FUNC}.csv

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_${Q_BITS_TEXT}_${GROUP_SIZE}gs_${SCALE_BITS}scale_2gs_64g.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_ppl_${Q_BITS_TEXT}_${GROUP_SIZE}gs_${SCALE_BITS}scale_2gs_64.csv

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_${Q_BITS_TEXT}_64_${GROUP_SIZE}gs_${SCALE_BITS}scale.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_ppl_${Q_BITS_TEXT}_64_${GROUP_SIZE}gs_${SCALE_BITS}scale.csv

# LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_loss_${Q_BITS_TEXT}_${GROUP_SIZE}gs_2_4_scale.csv
# PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_${METHOD}_ppl_${Q_BITS_TEXT}_${GROUP_SIZE}gs_2_4_scale.csv
N_PROC=1


CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=0 linear_sensitivity.py \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--method ${METHOD} \
--quant_model_paths "${QMODEL_PATHS[@]}" \
--quant_model_bits ${Q_BITS} \
--n_sample ${N_SAMPLE} \
--loss_csv_file ${LOSS_CSV_FILE} \
--ppl_csv_file ${PPL_CSV_FILE} \
--config ${CONFIG} \
--loss_func ${LOSS_FUNC}

# CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID 
# --eval_ppl \
# --eval_zeroshot \
# CUDA_DEVICE_ORDER=PCI_BUS_ID 