DEVICES=${1}

# MODEL_PATH=/SSD/JG/checkpoints/meta-llama
# MODEL_NAME=Llama-2-7b-hf

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf

# MODEL_PATH=facebook
# MODEL_NAME=opt-6.7b
# MODEL_NAME=opt-13b
# MODEL_NAME=opt-30b
# MODEL_NAME=opt-66b

FOLDER=/NAS/SJ/nsgaquant/latency_table

ENV=rtx6000ada
# ENV=a100
ITER=10
# LATENCY_FILE=${MODEL}_${ENV}.json
LATENCY_FILE=${MODEL_NAME}_${ENV}_token.json
# LATENCY_FILE=${MODEL}_${ENV}_prompt.json

CUDA_VISIBLE_DEVICES=${DEVICES} python -m latency_table \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--result_folder ${FOLDER} \
--result_file ${LATENCY_FILE} \
--iteration ${ITER} \
--generation

