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
# SMALL_GROUP_SIZE=64
SMALL_GROUP_SIZE=128
SMALL_AXIS=1
SMALL_QSCALE=false
SMALL_QZERO=false
SMALL_MODEL_PATH=/SSD/hqq/Llama-2-7b-hf_${SMALL_WBITS}bit_${SMALL_GROUP_SIZE}gs_${SMALL_AXIS}axis_qscale_${SMALL_QSCALE}_qzero_${SMALL_QZERO}

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

# QUANT_METHOD=owq
# SMALL_WBITS=2.1
# SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

# LARGE_WBITS=4.1
# LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth


OBJ=bits
PREFER="metric#2.0 bits#3.25"
EXPR_FOLDER=save/search

# EXPR_FILE=Llama-2-7b-hf_bits_loss_gptq_iter_300_nsga2_2_4_2410070911/iter_270.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_owq_iter_300_nsga2_2.1_4.1_2410071301/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_owq_iter_300_nsga2_2.01_4.01_2410071302/iter_300.stats
EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_2410071303/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_300_nsga2_2_4_2410051059/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_200_nsga2_2_4_2410051103/iter_200.stats
TARGET_BITS_RANGE="3.245 3.255"
SAVE=save/result/${TODAY}
N=10
DATASETS=wikitext2

CUDA_VISIBLE_DEVICES=${DEVICES} python post_search.py \
    --model_name ${MODEL_PATH}/${MODEL_NAME} \
    --config ${CONFIG} \
    --large_model_path ${LARGE_MODEL_PATH} \
    --large_model_bits ${LARGE_WBITS} \
    --small_model_path ${SMALL_MODEL_PATH} \
    --small_model_bits ${SMALL_WBITS} \
    --sec_obj ${OBJ} \
    -n ${N} \
    --save ${SAVE} \
    --expr ${EXPR_FOLDER}/${EXPR_FILE} \
    --prefer ${PREFER} \
    --datasets ${DATASETS} \
    --only_front False \
    --target_bits_range ${TARGET_BITS_RANGE} \
    --quant_method ${QUANT_METHOD}


    # --greedy_search_result_path ${GREEDY_SEARCH}
# GREEDY_SEARCH=''
# GREEDY_SEARCH=csv/greedy_search/Llama-2-7b-hf_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_2.495_2.505_2410050639/iter_100.stats
# TARGET_BITS_RANGE="2.495 2.505"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_2.995_3.005_2410050901/iter_100.stats
# TARGET_BITS_RANGE="2.995 3.005"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_3.495_3.505_2410050639/iter_100.stats
# TARGET_BITS_RANGE="3.495 3.505"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_2.19_2.21_2410041204/iter_100.stats
# TARGET_BITS_RANGE="2.19 2.21"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_2.39_2.41_2410041204/iter_100.stats
# TARGET_BITS_RANGE="2.39 2.41"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_2.59_2.61_2410041204/iter_100.stats
# TARGET_BITS_RANGE="2.59 2.61"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_2.79_2.81_2410041204/iter_100.stats
# TARGET_BITS_RANGE="2.79 2.81"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_3.19_3.21_2410041644/iter_100.stats
# TARGET_BITS_RANGE="3.19 3.21"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_3.39_3.41_2410041644/iter_100.stats
# TARGET_BITS_RANGE="3.39 3.41"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_3.59_3.61_2410041644/iter_100.stats
# TARGET_BITS_RANGE="3.59 3.61"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_3.79_3.81_2410041645/iter_100.stats
# TARGET_BITS_RANGE="3.79 3.81"
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_ga_2410032303/iter_100.stats # 2.99~3.01
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_nsga2_2410032317/iter_100.stats # 2.9~4.0
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_100_nsga2_2410032323/iter_100.stats # 2.9~3.5
# EXPR_FILE=Llama-2-7b-hf_bits_loss_2410021827/iter_200.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_2409292036/iter_400.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_2409291516/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_2409291301/iter_200.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_2409282026/iter_100.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_2409282026/iter_51.stats