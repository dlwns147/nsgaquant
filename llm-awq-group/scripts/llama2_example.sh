DEVICES=${1}


MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
MODEL_NAME=Meta-Llama-3-8B
CONFIG=/NAS/SJ/nsgaquant/config/llama_awq.json

# W_BIT=3
GROUP_SIZE=128
# AWQ_CACHE_BIT=2
# AWQ_CACHE=awq_cache/${MODEL_NAME}_w4_g${GROUP_SIZE}.pt
# AWQ_CACHE=awq_cache/${MODEL_NAME}_w${AWQ_CACHE_BIT}_g${GROUP_SIZE}.pt
# AWQ_CACHE=awq_cache/${MODEL_NAME}_w${AWQ_CACHE_BIT}_g128.pt

# AWQ_CACHE_BITS="2 4"
# AWQ_CACHE_BITS_TEXT="24"
# AWQ_CACHE_BITS="2 3"
# AWQ_CACHE_BITS_TEXT="23"
AWQ_CACHE_BITS="2 3 4"
AWQ_CACHE_BITS_TEXT="234"
AWQ_CACHE_LIST=()
for B in ${AWQ_CACHE_BITS}
do
    AWQ_CACHE_LIST+=( "/NAS/SJ/llm-awq-orig/awq_cache/${MODEL_NAME}_w${B}_g${GROUP_SIZE}.pt" )
done
# AWQ_CACHE_LIST+=( "/NAS/SJ/llm-awq/awq_cache/${MODEL_NAME}_w3_g${GROUP_SIZE}.pt" "/NAS/SJ/llm-awq/awq_cache/${MODEL_NAME}_w3_g${GROUP_SIZE}.pt" "/NAS/SJ/llm-awq/awq_cache/${MODEL_NAME}_w4_g${GROUP_SIZE}.pt")


# Q_BACKEND=real
# DUMP_QUANT=/SSD/awq/${MODEL_NAME}_w${W_BIT}_g${GROUP_SIZE}_real.pt

Q_BACKEND=fake
# DUMP_FAKE=/SSD/awq/${MODEL_NAME}_w${W_BIT}_g${GROUP_SIZE}_fake_${W_BIT}bit_awq.pt
# DUMP_FAKE=/SSD/awq/${MODEL_NAME}_w${W_BIT}_g${GROUP_SIZE}_fake_${AWQ_CACHE_BIT}bit_awq.pt
# DUMP_FAKE=/SSD/awq/${MODEL_NAME}_w${W_BIT}_g${GROUP_SIZE}_fake_${AWQ_CACHE_BIT}bit_128gs_awq.pt
DUMP_FAKE=/SSD/awq/${MODEL_NAME}_w234_g${GROUP_SIZE}_fake_group.pt

ARCH_FILE=/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_awq_bits_ppl_234_group_128gs_3scale_2_4_lp_1.0_1.0.json
# ARCH_FILE=/NAS/SJ/nsgaquant/save/result/2411080409_awq_2.995_3.005/results_arch.json

LOSS_CSV_FILE=/NAS/SJ/nsgaquant/csv/sensitivity/${MODEL_NAME}_awq_loss_${AWQ_CACHE_BITS_TEXT}_${GROUP_SIZE}_auto_scale.csv

# run AWQ search (optional; we provided the pre-computed results)
# CUDA_VISIBLE_DEVICES=${DEVICES} python -m awq.entry \
#     --model_path ${MODEL_PATH}/${MODEL_NAME} \
#     --w_bit ${AWQ_CACHE_BIT} \
#     --q_group_size ${GROUP_SIZE} \
#     --run_awq \
#     --dump_awq ${AWQ_CACHE}

# evaluate the AWQ quantize model (simulated pseudo quantization)
# CUDA_VISIBLE_DEVICES=${DEVICES} python -m awq.entry \
#     --model_path ${MODEL_PATH}/${MODEL_NAME} \
#     --tasks wikitext \
#     --w_bit ${W_BIT} --q_group_size ${GROUP_SIZE} \
#     --load_awq awq_cache/${MODEL_NAME}_w${W_BIT}_g${GROUP_SIZE}.pt \
#     --q_backend fake

# generate real quantized weights (w4)
# CUDA_VISIBLE_DEVICES=${DEVICES} python -m awq.entry \
#     --model_path ${MODEL_PATH}/${MODEL_NAME} \
#     --w_bit ${W_BIT} \
#     --q_group_size ${GROUP_SIZE} \
#     --load_awq ${AWQ_CACHE} \
#     --q_backend ${Q_BACKEND} \
#     --dump_fake ${DUMP_FAKE}
    # --dump_quant ${DUMP_QUANT}

CUDA_VISIBLE_DEVICES=${DEVICES} python -m awq.entry \
    --model_path ${MODEL_PATH}/${MODEL_NAME} \
    --config ${CONFIG} \
    --q_group_size ${GROUP_SIZE} \
    --q_backend ${Q_BACKEND} \
    --load_awq "${AWQ_CACHE_LIST[@]}" \
    --awq_bits ${AWQ_CACHE_BITS} \
    --loss_csv_file ${LOSS_CSV_FILE} \
    --tasks wikitext 
    # --arch_file ${ARCH_FILE}
    # --dump_fake ${DUMP_FAKE} \

# load and evaluate the real quantized model (smaller gpu memory usage)
# python -m awq.entry --model_path /dataset/llama2-hf/$MODEL \
#     --tasks wikitext \
#     --w_bit 4 --q_group_size 128 \
#     --load_quant quant_cache/$MODEL-w4-g128-awq.pt
