DEVICES=2
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_PATH=/SSD/.cache/
# MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

METHOD=awq

SEQLEN=2048
N_SAMPLE=128

OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

DATASETS=( "wikitext2" "c4" )

OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/${MODEL_NAME}-${METHOD}.csv
TARGET_BITS=( 3 )

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} get_algorithm_ppl_zeroshot.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--gpu_id ${DEVICES} \
--method ${METHOD} \
--seqlen ${SEQLEN} \
--n_sample ${N_SAMPLE} \
--eval_datasets ${DATASETS[@]} \
--zeroshot \
--output_path ${OUTPUT_PATH} \
--target_bits ${TARGET_BITS[@]} \

# --do_prune \
# --do_owq \
# --output_path ${OUTPUT_PATH} \
