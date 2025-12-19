### Install
```
pip install -r requirements.txt
```

### Train (cli)
```
# 1. Baseline
train.py --hf_token "your_hf_token"

# 2. Baseline+Target Modules
train.py --hf_token "your_hf_token" --target_modules 4 --save_model_path "./ex/qora-base-4"

# 3. alpha 16
train.py --hf_token "your_hf_token" --lora_alpha 16 --save_model_path "./ex/qora-alpha-16" 

# 4. alpha 32
train.py --hf_token "your_hf_token" --lora_alpha 32 --save_model_path "./ex/qora-alpha-32"

# 5. alpha 32 + dropout
train.py --hf_token "your_hf_token" --lora_alpha 32 --lora_dropout 0.5 --save_model_path "./ex/qora-alpha-32-dropout"

# 6. Final
train.py --hf_token "your_hf_token" --lora_alpha 32 --lora_dropout 0.5 --num_train_epochs 0.1 --save_model_path "./ex/qora-final"
```

### Evaluation
```
eval.py --hf_token "your_hf_token" --lora_path "./ex/ex_name"
```