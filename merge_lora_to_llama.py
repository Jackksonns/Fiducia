import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 路径设置
base_model_path = "models"  # 原始Llama2 7B Chat模型目录
adapter_path = "."          # LoRA adapter文件所在目录（即当前目录）
output_path = "merged_llama2_7bchat"  # 合并后输出目录

# 2. 加载原始模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.float16, local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, local_files_only=True
)

# 3. 加载LoRA Adapter
peft_model = PeftModel.from_pretrained(model, adapter_path)

# 4. 合并LoRA权重
merged_model = peft_model.merge_and_unload()

# 5. 保存合并后的模型和分词器
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"合并完成，已保存到 {output_path}")