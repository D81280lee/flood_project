# Flood_project

<div align="center">
  
# 

</div>Performing risk assessment and prediction tasks on flooding incidents in Lexington County utilizing Large Language Models

## Set up environment
We conduct experiments with CUDA 13.0 and Ubuntu 24.04

Create conda environment
```
conda create -n Flood python=3.13
conda activate Flood
```
Install the related packages.
```
pip install -r requirements_flood_classifier.txt
```
Updated requirements: pip install -U "bitsandbytes>=0.43" "transformers>=4.43" "peft>=0.11" "accelerate>=0.33" datasets scikit-learn
python -c "import torch; print(torch.cuda.is_available())" 
python -c "import bitsandbytes as bnb; import bitsandbytes.cuda_setup.main as m; print(m.get_cuda_lib_handle())"

## Dataset
Dataset format expected:
- A CSV/JSON file with at least two columns:
  * text_column (default: "summary")
  * label_column (default: "loss_category")
- Labels can be strings; they will be mapped to integers.

The Flood_data dataset has 2 columns: "summary" and "loss_category"
Data is divided into train and val, with 75% of data utilized for training and the remaining 25% used for evaluation. 

The loss_category has 4 possible categories: Minor, Major, Severe, Catastrophic, which will be used to categorize flooding events.

## Execution/Training
Run the following command to train an 11B Llama-3.2 model on flooding events in Lexington, SC to perform classification and prediction. This will run the train_lora_flood_classifier program, which will train the llm on the dataset. The infer_flood_classifier program will ONLY BE USED FOR FINAL TESTING (Also, make sure that you run huggingface-cli login on bash so that you have been granted access to the model): 
```
For me use case: This has worked
py ./train_lora_flood_classifier.py \
  --model_name_or_path meta-llama/Llama-3.2-11B \
  --train_file data/train.csv \
  --eval_file  data/val.csv \
  --text_column summary \
  --label_column loss_category \
  --output_dir outputs/llama32-11b-lora \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 256 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 \
  --do_train --do_eval \
  --label_list "Minor,Major,Severe,Catastrophic" \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --bf16


