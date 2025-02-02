DEVICES=
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_PATH=/SSD/.cache/
# MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

Q_BITS="2 3 4"
Q_BITS_TEXT="234"

# METHOD="hqq layer_prune"
# METHOD_TEXT="hqq_layer_prune"
METHOD=awq
METHOD_TEXT=hqq
GROUP_SIZE=128
AXIS=1
QSCALE=false
QZERO=false

# QMODEL_PATHS=()
# for B in ${Q_BITS}
# do
#     QMODEL_PATHS+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
# done
# QMODEL_PATHS=( "/SSD/hqq/${MODEL_NAME}_2bit_64gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_3bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_4bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}")
QMODEL_PATHS=( "/SSD/Woo/hqq/${MODEL_NAME}_2bit_${GROUP_SIZE}gs_${AXIS}axis" "/SSD/Woo/hqq/${MODEL_NAME}_3bit_${GROUP_SIZE}gs_${AXIS}axis" "/SSD/Woo/hqq/${MODEL_NAME}_4bit_${GROUP_SIZE}gs_${AXIS}axis")

N_OUTLIER=32
OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

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

OBJ=bits
TARGET_BITS=2.1

THRESHOLD=0.005
PREFER="metric#0.0 bits#${TARGET_BITS}"
EXPR_FOLDER=save/search

MIN_BITS=$(echo "scale=3; $TARGET_BITS - $THRESHOLD" | bc)
MAX_BITS=$(echo "scale=3; $TARGET_BITS + $THRESHOLD" | bc)


SAVE=save/result/${TODAY}_${MODEL_NAME}_${METHOD_TEXT}_${MIN_BITS}_${MAX_BITS}
N=1
DATASETS=( "wikitext2" "c4" )
# DATASETS=wikitext2
# DATASETS=c4
LATENCY_TABLE=/NAS/JG/QAS4SD/llama2_7b_lpe_24bit.json

OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/post_search_awq/${MODEL_NAME}-${METHOD}.csv


# evaluation부터 zeroshot task 실험 코드 세팅은 아래와 같음

DEVICES=3
MODEL_NAME=Llama-2-7b-hf
OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth
EXPR_FILE=quant/2501302006_Llama-2-7b-hf_bits_loss_hqq_layer_prune_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample/iter_300.stats
OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/post_search_awq_whether_prune/${MODEL_NAME}-${METHOD}.csv

#DEVICES=1
#MODEL_NAME=Llama-2-13b-hf
#OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth
#EXPR_FILE=quant/2501302006_Llama-2-13b-hf_bits_loss_hqq_layer_prune_iter_400_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample/iter_400.stats
#OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/post_search_awq_whether_prune/${MODEL_NAME}-${METHOD}.csv


N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} post_search_woo.py \
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
--prefer ${PREFER} \
--datasets ${DATASETS[@]} \
--sec_obj_range ${MIN_BITS} ${MAX_BITS} \
--method ${METHOD} \
--zeroshot \
--output_path ${OUTPUT_PATH} \
--do_prune \
# --outlier_path ${OUTLIER_PATH} \
# --do_owq \
# --latency_table_file ${LATENCY_TABLE}
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
