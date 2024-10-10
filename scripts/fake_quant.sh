DEVICES=${1}
TODAY=`date +%y%m%d%H%M`

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf

SAVE=save/result/${TODAY}
DATASET="wikitext2"

# ACT_QUANT='per_token'
# ACT_QUANT='per_tensor'
ACT_QUANT='per_tensor_la'
# ACT_QUANT='per_tensor_smooth'

W_BITS=4
A_BITS=8

CUDA_VISIBLE_DEVICES=${DEVICES} python fake_quant.py \
    --model_name ${MODEL_PATH}/${MODEL_NAME} \
    --save ${SAVE} \
    --dataset ${DATASET} \
    --act_quant ${ACT_QUANT} \
    --w_bits ${W_BITS} \
    --a_bits ${A_BITS}
