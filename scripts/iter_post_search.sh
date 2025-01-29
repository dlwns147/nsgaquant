DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

Q_BITS="2 3 4"
Q_BITS_TEXT="234"

# METHOD="hqq layer_prune"
# METHOD_TEXT="hqq_layer_prune"
METHOD=hqq
METHOD_TEXT=hqq
GROUP_SIZE=128
AXIS=1
QSCALE=false
QZERO=false

QMODEL_PATHS=()
for B in ${Q_BITS}
do
    QMODEL_PATHS+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
done
# QMODEL_PATHS=( "/SSD/hqq/${MODEL_NAME}_2bit_64gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_3bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_4bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}")

N_OUTLIER=32
OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

OBJ=bits

EXPR_FOLDER=save/search/quant

EXPR_FILE=2501231758_Llama-2-13b-hf_bits_loss_hqq_iter_400_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample_outlier/iter_400.stats
# EXPR_FILE=2501231721_Llama-2-13b-hf_bits_loss_hqq_iter_400_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample/iter_400.stats
# EXPR_FILE=2501231757_Llama-2-13b-hf_bits_loss_hqq_iter_400_234_obj_2_4_cross_entropy_co_0.9_mut_0.1_wikitext2_128sample_outlier/iter_400.stats
# EXPR_FILE=2501231720_Llama-2-13b-hf_bits_loss_hqq_iter_400_234_obj_2_4_cross_entropy_co_0.9_mut_0.1_wikitext2_128sample/iter_400.stats


# EXPR_FILE=2501231756_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample_outlier/iter_300.stats
# EXPR_FILE=2501231752_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_cross_entropy_co_0.9_mut_0.1_wikitext2_128sample_outlier/iter_300.stats
# EXPR_FILE=2501231719_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample/iter_300.stats
# EXPR_FILE=2501221949_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_cross_entropy_co_0.9_mut_0.1_wikitext2_128sample/iter_300.stats

# EXPR_FILE=2501221949_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_cross_entropy_co_0.9_mut_0.1_wikitext2_128sample/iter_300.stats
# EXPR_FILE=2501191909_Llama-2-13b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_lp_1.0_1.0_wikitext2_64sample/iter_300.stats
# EXPR_FILE=2501191909_Llama-2-13b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_lp_1.0_1.0_wikitext2_32sample/iter_300.stats
# EXPR_FILE=2501191908_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_lp_1.0_1.0_wikitext2_64sample/iter_300.stats
# EXPR_FILE=2501191906_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_lp_1.0_1.0_wikitext2_32sample/iter_300.stats

# EXPR_FILE=2501072019_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_lp_1.0_1.0_c4_16sample/iter_200.stats
# EXPR_FILE=2501072019_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_lp_1.0_1.0_c4_16sample/iter_300.stats
# EXPR_FILE=2501051827_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_lp_1.0_1.0_c4_16sample/iter_300.stats


N=1
DATASETS="wikitext2 c4"
LATENCY_TABLE=/NAS/JG/QAS4SD/llama2_7b_lpe_24bit.json

TARGET_SEC_OBJ_LIST=(2.25 2.5 2.75 3.0 3.25 3.5 3.75)
# TARGET_SEC_OBJ_LIST=(3.01)
SEC_OBJ_THRESHOLD=0.005

for TARGET_SEC_OBJ in ${TARGET_SEC_OBJ_LIST[*]}
do
    MIN_SEC_OBJ=$(echo "scale=3; $TARGET_SEC_OBJ - $SEC_OBJ_THRESHOLD" | bc)
    MAX_SEC_OBJ=$(echo "scale=3; $TARGET_SEC_OBJ + $SEC_OBJ_THRESHOLD" | bc)
    SAVE=save/result/${TODAY}_${MODEL_NAME}_${OBJ}_${MIN_SEC_OBJ}_${MAX_SEC_OBJ}

    PREFER="metric#0 ${OBJ}#${TARGET_SEC_OBJ}"

    N_PROC=1
    CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} post_search.py \
    --gpu_id ${DEVICES} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --quant_model_paths "${QMODEL_PATHS[@]}" \
    --quant_model_bits ${Q_BITS} \
    --sec_obj ${OBJ} \
    -n ${N} \
    --save ${SAVE} \
    --debug \
    --expr ${EXPR_FOLDER}/${EXPR_FILE} \
    --datasets ${DATASETS} \
    --sec_obj_range ${MIN_SEC_OBJ} ${MAX_SEC_OBJ} \
    --method ${METHOD} \
    --outlier_path ${OUTLIER_PATH} \
    --zeroshot \
    --high_tradeoff
    # --prefer ${PREFER} \
    # --outlier_path ${OUTLIER_PATH} \
    # --latency_table_file ${LATENCY_TABLE}
    # --only_front \
done

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
