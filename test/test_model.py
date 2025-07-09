import torch
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

def generate_response(prompt, model, tokenizer, max_length=512):
    """生成回复"""
    # 构建输入格式
    input_text = f"Human: {prompt}\n\nAssistant: "
    
    # 对输入进行编码
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取助手的回复
    response = response.split("Assistant: ")[-1].strip()
    return response

def main():
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model_path = "./my_trained_model"
    
    # 加载基础模型配置
    config = PeftConfig.from_pretrained(model_path)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载LoRA模型
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 测试模型
    test_prompts = [
        "你好，请介绍一下你自己",
        "你能帮我写一首诗吗？",
        "解释一下什么是人工智能"
    ]
    
    print("\n开始测试模型...")
    for prompt in test_prompts:
        print(f"\n用户: {prompt}")
        response = generate_response(prompt, model, tokenizer)
        print(f"助手: {response}")

if __name__ == "__main__":
    main() 