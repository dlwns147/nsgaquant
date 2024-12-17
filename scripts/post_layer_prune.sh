DEVICES=${1}
TODAY=`date +%y%m%d%H%M`

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
MODEL_NAME=Llama-2-70b-hf

CONFIG=config/llama.json

METHOD=layer_prune
Q_BITS=16

# OBJ=sparsity
OBJ=params
# OBJ=latency
EXPR_FOLDER=save/search


TARGET_SEC_OBJ=0.5
SEC_OBJ_THRESHOLD=0.005

MIN_SEC_OBJ=$(echo "scale=3; $TARGET_SEC_OBJ - $SEC_OBJ_THRESHOLD" | bc)
MAX_SEC_OBJ=$(echo "scale=3; $TARGET_SEC_OBJ + $SEC_OBJ_THRESHOLD" | bc)

# REMAINED_LAYER=55
# NUM_LAYER=64
# THRESHOLD=0.001
# MIN_SEC_OBJ=$(echo "scale=3; $REMAINED_LAYER / $NUM_LAYER - $THRESHOLD" | bc)
# MAX_SEC_OBJ=$(echo "scale=3; $REMAINED_LAYER / $NUM_LAYER + $THRESHOLD" | bc)

PREFER="metric#0 ${OBJ}#${TARGET_SEC_OBJ}"

EXPR_FILE=2412162219_Llama-2-70b-hf_sparsity_loss_layer_prune_iter_320_n_iter_80_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.40_1.0_128sample/iter_320.stats
# EXPR_FILE=2412160850_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.40_1.0_128sample_pass_ratio_0.1/iter_128.stats
# EXPR_FILE=2412161012_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.40_1.0_128sample_pass_ratio_0.1/iter_160.stats
# EXPR_FILE=2412152121_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.40_1.0_128sample_pass_ratio_0.1/iter_128.stats


# EXPR_FILE=2412151247_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.45_1.0_128sample_pass_ratio_0.1/iter_128.stats
# EXPR_FILE=2412151140_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.4_1.0_128sample_pass_ratio_0.1/iter_64.stats
# EXPR_FILE=2412151129_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1._jsd_mut_0.1_mask_0.4_1.0_128sample_pass_ratio_0.1/iter_128.stats
# EXPR_FILE=2412131152_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1._jsd_mut_0.1_mask_0.4_1.0_64sample/iter_128.stats
# EXPR_FILE=2412112057_Llama-2-7b-hf_latency_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1e9_jsd_mut_0.1_layer_prune_0.4_1.0/iter_128.stats

# EXPR_FILE=2412091639_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_160_n_iter_32_nsga2_obj_0.4_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_128.stats
# EXPR_FILE=2412091008_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.4_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_160.stats
# EXPR_FILE=2412090938_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.4_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_128.stats

# EXPR_FILE=2412090937_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_128.stats
# EXPR_FILE=2412090937_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_96.stats
# EXPR_FILE=2412090031_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_160.stats
# EXPR_FILE=2411131721_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_nsga2_obj_0.5_1.0_jsd_mut_0.1_layer_prune_0.5_1.0/iter_127.stats
# EXPR_FILE=2411251843_Llama-2-13b-hf_params_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_159.stats
# EXPR_FILE=2411182009_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_127.stats

# EXPR_FILE=2411181806_Llama-2-7b-hf_params_loss_layer_prune_iter_64_n_iter_32_nsga2_obj_0.001_1._jsd_mut_0.1_layer_prune_0.5_1.0/iter_63.stats
# EXPR_FILE=2411131842_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_nsga2_obj_0.5_1.0_jsd_mut_0.05_layer_prune_0.5_1.0/iter_127.stats
# EXPR_FILE=2411131757_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_nsga2_obj_0.5_1.0_jsd_mut_0.1_layer_prune_0.5_1.0/iter_127.stats
# EXPR_FILE=2411131712_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_nsga2_obj_0.5_1.0_jsd_mut_0.05_layer_prune_0.5_1.0/iter_127.stats
# EXPR_FILE=2411131721_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_nsga2_obj_0.5_1.0_jsd_mut_0.1_layer_prune_0.5_1.0/iter_127.stats
# EXPR_FILE=2411131308_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_nsga2__obj_0.5_1.0_jsd_mut_0.05_layer_prune_0.5_1.0/iter_63.stats

LATENCY_TABLE=latency_table/${MODEL_NAME}_rtx6000ada.json
SAVE=save/result/${TODAY}_${MODEL_NAME}_${METHOD}
N=1
DATASETS="wikitext2 c4"
# DATASETS=c4

CUDA_VISIBLE_DEVICES=${DEVICES} python post_search.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--quant_model_bits ${Q_BITS} \
--sec_obj ${OBJ} \
-n ${N} \
--save ${SAVE} \
--debug \
--expr ${EXPR_FOLDER}/${EXPR_FILE} \
--datasets ${DATASETS} \
--method ${METHOD} \
--prefer ${PREFER} \
--sec_obj_range ${MIN_SEC_OBJ} ${MAX_SEC_OBJ} \
--zeroshot
# --latency_table_file ${LATENCY_TABLE} \
# --latency \
# --only_front \\
