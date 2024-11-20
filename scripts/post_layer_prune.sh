DEVICES=${1}
TODAY=`date +%y%m%d%H%M`

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=config/llama.json

METHOD=layer_prune
Q_BITS=16

# OBJ=sparsity
OBJ=params
EXPR_FOLDER=save/search

EXPR_FILE=2411182009_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_127.stats
# EXPR_FILE=2411181806_Llama-2-7b-hf_params_loss_layer_prune_iter_64_n_iter_32_nsga2_obj_0.001_1._jsd_mut_0.1_layer_prune_0.5_1.0/iter_63.stats
# EXPR_FILE=2411131842_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_nsga2_obj_0.5_1.0_jsd_mut_0.05_layer_prune_0.5_1.0/iter_127.stats
# EXPR_FILE=2411131757_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_nsga2_obj_0.5_1.0_jsd_mut_0.1_layer_prune_0.5_1.0/iter_127.stats
# EXPR_FILE=2411131712_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_nsga2_obj_0.5_1.0_jsd_mut_0.05_layer_prune_0.5_1.0/iter_127.stats
# EXPR_FILE=2411131721_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_nsga2_obj_0.5_1.0_jsd_mut_0.1_layer_prune_0.5_1.0/iter_127.stats
# EXPR_FILE=2411131308_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_nsga2__obj_0.5_1.0_jsd_mut_0.05_layer_prune_0.5_1.0/iter_63.stats

SAVE=save/result/${TODAY}_${METHOD}
N=5
DATASETS=wikitext2

CUDA_VISIBLE_DEVICES=${DEVICES} python post_search.py \
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
    --only_front \

