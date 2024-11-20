from huggingface_hub import snapshot_download
model_path = "meta-llama"
model_name = "Llama-2-7b-hf"
# model_name = "Meta-Llama-3-8B"
print(f'model_path : {model_path}, model_name : {model_name}')
snapshot_download(repo_id=f'{model_path}/{model_name}', local_dir=f"/SSD/huggingface/{model_path}/{model_name}")