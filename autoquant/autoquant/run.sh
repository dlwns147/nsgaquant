devices="1"
method="awq"
model_name="meta-llama/Llama-2-7b-hf"
arch_path="/NAS/Woo/Automation/autoopt/archs/post_search/7b_owq/results_arch.json"
arch_idx=0
owq_path="/NAS/SJ/nsgaquant/outlier/Llama-2-7b-hf/w16_r32/outlier.pth"
seed=0
result_save_name="trash"

CUDA_VISIBLE_DEVICES=$devices python main.py \
    $method \
    --model_name $model_name \
    --arch_path $arch_path \
    --arch_idx $arch_idx \
    --owq_path $owq_path \
    --eval \
    --seed $seed \
    --result_save_name $result_save_name