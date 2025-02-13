DEVICES=3
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

# DATASETS=( "wikitext2" "c4" )
DATASETS=( )

# OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/${MODEL_NAME}-${METHOD}.csv
GROUP_SIZE=128
TARGET_BITS=( 3 )

# ARCH_PATH=/NAS/Woo/Automation/autoopt/archs/HQQ_woPrior_random_linear_wINT3_meta-llama_Llama-2-7b-hf.json
# ARCH_PATH=/NAS/Woo/Automation/autoopt/archs/HQQ_woPrior_random_linear_wINT3_meta-llama_Llama-2-13b-hf.json
# OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/awq_random_sample/${MODEL_NAME}-${METHOD}.csv
OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/group_size_${GROUP_SIZE}/${MODEL_NAME}-${METHOD}_asym.csv

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} get_algorithm_ppl_zeroshot_boolq.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--gpu_id ${DEVICES} \
--method ${METHOD} \
--seqlen ${SEQLEN} \
--n_sample ${N_SAMPLE} \
--output_path ${OUTPUT_PATH} \
--target_bits ${TARGET_BITS[@]} \
--group_size ${GROUP_SIZE} \
--output_path ${OUTPUT_PATH} \
--zeroshot
# --do_clip_sym

OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/group_size_${GROUP_SIZE}/${MODEL_NAME}-${METHOD}_sym.csv

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} get_algorithm_ppl_zeroshot_boolq.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--gpu_id ${DEVICES} \
--method ${METHOD} \
--seqlen ${SEQLEN} \
--n_sample ${N_SAMPLE} \
--output_path ${OUTPUT_PATH} \
--target_bits ${TARGET_BITS[@]} \
--group_size ${GROUP_SIZE} \
--output_path ${OUTPUT_PATH} \
--zeroshot \
--do_clip_sym

METHOD=gptq

OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/group_size_${GROUP_SIZE}/${MODEL_NAME}-${METHOD}.csv

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} get_algorithm_ppl_zeroshot_boolq.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--gpu_id ${DEVICES} \
--method ${METHOD} \
--seqlen ${SEQLEN} \
--n_sample ${N_SAMPLE} \
--output_path ${OUTPUT_PATH} \
--target_bits ${TARGET_BITS[@]} \
--group_size ${GROUP_SIZE} \
--output_path ${OUTPUT_PATH} \
--zeroshot
# --do_clip_sym

MODEL_NAME=Llama-2-13b-hf
METHOD=awq

OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/group_size_${GROUP_SIZE}/${MODEL_NAME}-${METHOD}_asym.csv

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} get_algorithm_ppl_zeroshot_boolq.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--gpu_id ${DEVICES} \
--method ${METHOD} \
--seqlen ${SEQLEN} \
--n_sample ${N_SAMPLE} \
--output_path ${OUTPUT_PATH} \
--target_bits ${TARGET_BITS[@]} \
--group_size ${GROUP_SIZE} \
--output_path ${OUTPUT_PATH} \
--zeroshot
# --do_clip_sym

OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/group_size_${GROUP_SIZE}/${MODEL_NAME}-${METHOD}_sym.csv

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} get_algorithm_ppl_zeroshot_boolq.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--gpu_id ${DEVICES} \
--method ${METHOD} \
--seqlen ${SEQLEN} \
--n_sample ${N_SAMPLE} \
--output_path ${OUTPUT_PATH} \
--target_bits ${TARGET_BITS[@]} \
--group_size ${GROUP_SIZE} \
--output_path ${OUTPUT_PATH} \
--zeroshot \
--do_clip_sym

METHOD=gptq
OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/group_size_${GROUP_SIZE}/${MODEL_NAME}-${METHOD}.csv

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} get_algorithm_ppl_zeroshot_boolq.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--gpu_id ${DEVICES} \
--method ${METHOD} \
--seqlen ${SEQLEN} \
--n_sample ${N_SAMPLE} \
--output_path ${OUTPUT_PATH} \
--target_bits ${TARGET_BITS[@]} \
--group_size ${GROUP_SIZE} \
--output_path ${OUTPUT_PATH} \
--zeroshot
# --do_clip_sym