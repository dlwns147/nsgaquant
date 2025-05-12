DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json
DTYPE=float16

Q_BITS="2 3 4"
Q_BITS_TEXT="234"

# METHOD="hqq layer_prune"
# METHOD=hqq
METHOD=awq
# METHOD=gptq
# METHOD="awq layer_prune"

GROUP_SIZE=128
AXIS=1
QSCALE=false
QZERO=false

QMODEL_PATHS_LIST=()
for B in ${Q_BITS}
do
    # QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
# QMODEL_PATHS=( "/SSD/hqq/${MODEL_NAME}_2bit_64gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_3bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_4bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}")
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

N_OUTLIER=32
OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

COMP_OBJ=bits
COMP_OBJ_TEXT=bits
TARGET_COMP_OBJ_VAL=3.0
# TARGET_COMP_OBJ_VAL=2.0

TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq openbookqa social_iqa"
ZEROSHOT_BATCH_SIZE=16

TARGET_COMP_OBJ=bits
COMP_OBJ_THRESHOLD=0.005
PREFER="metric#0.0 ${TARGET_COMP_OBJ}#${TARGET_COMP_OBJ_VAL}"

MIN_COMP_OBJ=$(echo "scale=3; $TARGET_COMP_OBJ_VAL - $COMP_OBJ_THRESHOLD" | bc)
MAX_COMP_OBJ=$(echo "scale=3; $TARGET_COMP_OBJ_VAL + $COMP_OBJ_THRESHOLD" | bc)

EXPR_FOLDER=save/search/quant

EXPR_FILE=2504100856_Llama-2-7b-hf_bits_loss_hqq_iter_100_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_32sample_rbf/iter_100.stats
# EXPR_FILE=2502101608_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4.1_jsd_co_0.9_mut_0.1_wikitext2_32sample_rbf_outlier_234_mixed/iter_200.stats
# EXPR_FILE=2502012035_Llama-2-7b-hf_bits_loss_hqq_layer_prune_iter_300_234_obj_1.99_4_jsd_co_0.9_mut_0.1_wikitext2_32sample_lp_0.001_1.0/iter_300.stats
# EXPR_FILE=2501231721_Llama-2-13b-hf_bits_loss_hqq_iter_400_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample/iter_400.stats
# EXPR_FILE=2501231719_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample/iter_300.stats
# EXPR_FILE=2501231756_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample_outlier/iter_300.stats
# EXPR_FILE=2411211754_Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_1.0_1.0/iter_299.stats

SAVE=save/result/${TODAY}_${MODEL_NAME}_${COMP_OBJ}_${MIN_COMP_OBJ}_${MAX_COMP_OBJ}
N=1
DATASETS="wikitext2 c4"

N_PROC=1
# N_PROC=2
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} post_search.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${MIN_COMP_OBJ[*]} \
--comp_obj_max ${MAX_COMP_OBJ[*]} \
--quant_model_paths ${QMODEL_PATHS} \
--quant_model_bits ${Q_BITS} \
--group_size ${GROUP_SIZE} \
-n ${N} \
--save ${SAVE} \
--debug \
--expr ${EXPR_FOLDER}/${EXPR_FILE} \
--prefer ${PREFER} \
--datasets ${DATASETS} \
--method ${METHOD} \
--zeroshot \
--tasks ${TASKS} \
--zeroshot_batch_size ${ZEROSHOT_BATCH_SIZE}
# --latency_table_file ${LATENCY_TABLE}
# --outlier_path ${OUTLIER_PATH} \
# --only_front \


    # --greedy_search_result_path ${GREEDY_SEARCH}
# GREEDY_SEARCH=''
# GREEDY_SEARCH=csv/greedy_search/Llama-2-7b-hf_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_layer_prune_iter_300_nsga2_2_4_obj_2_4_mut_0.1_layer_prune_0.95_1.0_2410311536/iter_299.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_awq_iter_300_nsga2_2_4_0.01_2410211524/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_mut_prob_0.1_2410101147/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_mut_prob_0.2_2410101159/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_mut_prob_0.02_2410101352/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_gptq_iter_300_nsga2_2_4_2410070911/iter_270.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_owq_iter_300_nsga2_2.1_4.1_2410071301/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_owq_iter_300_nsga2_2.01_4.01_2410071302/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_2410071303/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_300_nsga2_2_4_2410051059/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_200_nsga2_2_4_2410051103/iter_200.stats
# TARGET_BITS_RANGE="${MIN_BITS} ${MAX_BITS}"
# QMODEL_PATHS=("/SSD/awq/${MODEL_NAME}_w2_g64_fake_${SCALE_BITS}bit_128gs_awq.pt" "/SSD/awq/${MODEL_NAME}_w3_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt" "/SSD/awq/${MODEL_NAME}_w4_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt")

# METHOD=owq
# SMALL_WBITS=2.1
# SMALL_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${SMALL_WBITS}_wikitext2_fake.pth

# LARGE_WBITS=4.1
# LARGE_MODEL_PATH=/SSD/owq/${MODEL_NAME}_${LARGE_WBITS}_wikitext2.pth

# METHOD=awq
# METHOD_TEXT=awq
# GROUP_SIZE=128
# SCALE_BITS=2
# # SCALE_BITS=3

# QMODEL_PATHS=()
# for B in ${Q_BITS}
# do
#     # QMODEL_PATHS+=( "/SSD/awq/${MODEL_NAME}_w${B}_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt" )
#     QMODEL_PATHS+=( "/SSD/awq/${MODEL_NAME}_w${B}_g${GROUP_SIZE}_fake_${SCALE_BITS}scale_asym.pt" )
# done
