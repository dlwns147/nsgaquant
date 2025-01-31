DEVICES=${1}

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf

CONFIG=config/llama.json

METHOD=layer_prune
Q_BITS=16

# OBJ=sparsity
# OBJ=params
# OBJ=latency

# EXPR_FOLDER=save/search
EXPR_FOLDER=save/search/prune

COMP_OBJ="params"
# COMP_OBJ="sparsity"
# COMP_OBJ="params latency"
# TRADEOFF_OBJ="latency"

EXPR_FILE=2501301838_Llama-2-13b-hf_mo_params_latency_loss_iter_240_n_iter_40_obj_0.45_0.45_1_1_jsd_co_0.9_mut_0.1_mask_32sample_pop_200_wikitext2_rtx6000ada/iter_240.stats
# EXPR_FILE=2501301838_Llama-2-13b-hf_mo_params_latency_loss_iter_240_n_iter_40_obj_0.45_0.45_1_1_cross_entropy_co_0.9_mut_0.1_mask_32sample_pop_200_wikitext2_rtx6000ada/iter_240.stats
# EXPR_FILE=2501301733_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_240_n_iter_40_obj_0.45_1_cross_entropy_co_0.9_mut_0.1_mask_0.45_1.0_32sample_pop_200_wikitext2_rtx6000ada/iter_240.stats
# EXPR_FILE=2501301733_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_240_n_iter_40_obj_0.45_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_32sample_pop_200_wikitext2_rtx6000ada/iter_240.stats

# EXPR_FILE=2501301833_Llama-2-7b-hf_mo_params_latency_loss_iter_192_n_iter_32_obj_0.45_0.45_1_1_jsd_co_0.9_mut_0.1_mask_32sample_pop_200_wikitext2_rtx6000ada/iter_192.stats
# EXPR_FILE=2501301836_Llama-2-7b-hf_mo_params_latency_loss_iter_192_n_iter_32_obj_0.45_0.45_1_1_cross_entropy_co_0.9_mut_0.1_mask_32sample_pop_200_wikitext2_rtx6000ada/iter_192.stats
# EXPR_FILE=2501301734_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_192_n_iter_32_obj_0.45_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_32sample_pop_200_wikitext2_rtx6000ada/iter_192.stats
# EXPR_FILE=2501301734_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_192_n_iter_32_obj_0.45_1_cross_entropy_co_0.9_mut_0.1_mask_0.45_1.0_32sample_pop_200_wikitext2_rtx6000ada/iter_192.stats

# EXPR_FILE=2501282048_Llama-2-13b-hf_mo_params_latency_loss_iter_160_n_iter_40_obj_0.45_0.45_1_1_jsd_co_0.9_mut_0.1_mask_128sample_pop_40_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501282047_Llama-2-13b-hf_mo_params_latency_loss_iter_160_n_iter_40_obj_0.45_0.45_1_1_cross_entropy_co_0.9_mut_0.1_mask_128sample_pop_40_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501282045_Llama-2-7b-hf_mo_params_latency_loss_iter_128_n_iter_32_obj_0.45_0.45_1_1_jsd_co_0.9_mut_0.1_mask_128sample_pop_32_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501282046_Llama-2-7b-hf_mo_params_latency_loss_iter_128_n_iter_32_obj_0.45_0.45_1_1_cross_entropy_co_0.9_mut_0.1_mask_128sample_pop_32_wikitext2_rtx6000ada/iter_128.stats

# EXPR_FILE=2501251838_Llama-2-7b-hf_mo_params_latency_loss_iter_128_n_iter_32_obj_0.45_0.45_1_1_jsd_co_0.9_mut_0.1_mask_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501241959_Llama-2-13b-hf_mo_params_latency_loss_iter_160_n_iter_40_obj_0.45_0.45_1_1_cross_entropy_co_0.9_mut_0.1_mask_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501241954_Llama-2-13b-hf_mo_params_latency_loss_iter_160_n_iter_40_obj_0.45_0.45_1_1_jsd_co_0.9_mut_0.1_mask_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501241937_Llama-2-7b-hf_mo_params_latency_loss_iter_128_n_iter_32_obj_0.45_0.45_1_1_jsd_co_0.9_mut_0.1_mask_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501232003_Llama-2-13b-hf_mo_params_latency_loss_iter_160_n_iter_40_obj_0.45_0.01_1_1_cross_entropy_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501232002_Llama-2-7b-hf_mo_params_latency_loss_iter_128_n_iter_32_obj_0.45_0.01_1_1_cross_entropy_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats

ENV=rtx6000ada
# ENV=a100
LATENCY_TABLE=latency_table/${MODEL_NAME}_${ENV}_token.json
N=1
DATASETS="wikitext2 c4"

# TARGET_COMP_OBJ_LIST=(0.9 0.8 0.7 0.6 0.5)
TARGET_COMP_OBJ_LIST=(0.5 0.6 0.7 0.8 0.9)
# TARGET_COMP_OBJ_LIST=(0.5)
COMP_OBJ_THRESHOLD=0.005

for TARGET_COMP_OBJ in ${TARGET_COMP_OBJ_LIST[*]}
do
    # MIN_COMP_OBJ=($(echo "scale=3; $TARGET_COMP_OBJ - $COMP_OBJ_THRESHOLD" | bc) 0)
    # MAX_COMP_OBJ=($(echo "scale=3; $TARGET_COMP_OBJ + $COMP_OBJ_THRESHOLD" | bc) 1)
    MIN_COMP_OBJ=$(echo "scale=3; $TARGET_COMP_OBJ - $COMP_OBJ_THRESHOLD" | bc)
    MAX_COMP_OBJ=$(echo "scale=3; $TARGET_COMP_OBJ + $COMP_OBJ_THRESHOLD" | bc)
    TODAY=`date +%y%m%d%H%M`
    SAVE=save/result/${TODAY}_${MODEL_NAME}_${MIN_COMP_OBJ}_${MAX_COMP_OBJ}_${METHOD}

    PREFER="metric#0 ${COMP_OBJ}#${TARGET_COMP_OBJ}"

    CUDA_VISIBLE_DEVICES=${DEVICES} python post_search.py \
    --gpu_id ${DEVICES} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --quant_model_bits ${Q_BITS} \
    -n ${N} \
    --save ${SAVE} \
    --debug \
    --expr ${EXPR_FOLDER}/${EXPR_FILE} \
    --method ${METHOD} \
    --comp_obj ${COMP_OBJ} \
    --comp_obj_min ${MIN_COMP_OBJ[*]} \
    --comp_obj_max ${MAX_COMP_OBJ[*]} \
    --latency_table_file ${LATENCY_TABLE} \
    --datasets ${DATASETS} \
    --zeroshot \
    --latency \
    --prefer ${PREFER}
    # --high_tradeoff ${TRADEOFF_OBJ}
    # --high_tradeoff
done
# --latency \
# --only_front \\

# EXPR_FILE=2501221252_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.45_1_cross_entropy_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_32_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501211811_Llama-2-13b-hf_params_loss_layer_prune_iter_160_n_iter_40_obj_0.45_1_cross_entropy_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_40_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501211755_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_cross_entropy_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_40_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501211811_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_obj_0.45_1_cross_entropy_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_32_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501211748_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_cross_entropy_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_32_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501211131_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_cross_entropy_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501211117_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_cross_entropy_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats

# EXPR_FILE=2501161843_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_64sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501161843_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_32sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501161842_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_32sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501161842_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_64sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501141714_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_64sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501141713_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_32sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501141713_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_64sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501141713_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_32sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501131631_Llama-2-13b-hf_latency_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501131610_Llama-2-7b-hf_latency_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501130926_Llama-2-13b-hf_params_loss_layer_prune_iter_160_n_iter_40_obj_0.45_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501130915_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.45_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501131038_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_obj_0.45_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501130915_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.45_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501121401_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501121245_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501121402_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501121244_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501111816_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501111815_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501111728_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501111724_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats

# EXPR_FILE=2501111211_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501111209_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501071857_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1.0_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2412090938_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.4_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_128.stats

# EXPR_FILE=2501091409_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.45_0.95_jsd_co_0.9_mut_0.1_mask_0.45_0.95_32sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501091328_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.45_0.95_jsd_co_0.9_mut_0.1_mask_0.45_0.95_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501091327_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.45_0.95_jsd_co_1.0_mut_0.1_mask_0.45_0.95_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501091139_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.001_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_32sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501081743_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.001_1_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501081449_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1.0_jsd_co_1.0_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501081329_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1.0_jsd_co_1.0_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501072024_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1.0_jsd_co_1.0_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501072023_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1.0_jsd_co_1.0_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501071817_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.4_1.0_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501071857_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.4_1.0_jsd_co_0.9_mut_0.1_mask_0.4_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501061815_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_obj_0.45_1.0_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501061814_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_obj_0.45_1.0_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501051802_Llama-2-13b-hf_latency_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.45_1.0_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501051808_Llama-2-7b-hf_latency_loss_layer_prune_iter_128_n_iter_32_obj_0.45_1.0_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_200_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501051703_Llama-2-13b-hf_params_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.45_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_120_wikitext2_rtx6000ada/iter_160.stats
# EXPR_FILE=2501051704_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.45_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_96_wikitext2_rtx6000ada/iter_128.stats
# EXPR_FILE=2501042305_Llama-2-13b-hf_params_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.001_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_120_rtx6000ada/iter_160.stats
# EXPR_FILE=2501042305_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_96_rtx6000ada/iter_108.stats
# EXPR_FILE=2501031246_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_co_0.9_mut_0.1_mask_0.45_1.0_128sample_pop_96_rtx6000ada/iter_128.stats
# EXPR_FILE=2501030801_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.001_1_jsd_co_1.0_mut_0.1_mask_0.40_1.0_128sample_pop_120_rtx6000ada/iter_160.stats
# EXPR_FILE=2501030800_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_co_1.0_mut_0.1_mask_0.40_1.0_128sample_pop_96_rtx6000ada/iter_128.stats
# EXPR_FILE=2501021415_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_co_1.0_mut_0.1_mask_0.40_1.0_128sample_pop_64_rtx6000ada/iter_84.stats

# EXPR_FILE=2412301247_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.40_1.0_32sample_rtx6000ada/iter_128.stats
# EXPR_FILE=2412301247_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.40_1.0_64sample_rtx6000ada/iter_128.stats
# EXPR_FILE=2412301248_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.40_1.0_32sample_rtx6000ada/iter_160.stats
# EXPR_FILE=2412301249_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.001_1_jsd_mut_0.1_mask_0.40_1.0_64sample_rtx6000ada/iter_160.stats

