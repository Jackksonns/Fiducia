from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
import os
import torch

# 设置内存优化配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置本地模型路径
MODEL_PATH = "./model"

# 检查模型文件是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件未找到！请确保模型文件已下载到 {MODEL_PATH} 目录下")

print(f"正在从本地加载模型: {MODEL_PATH}")

try:
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 配置8位量化
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    # 使用8位量化加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        torch_dtype=torch.float16  # 使用半精度浮点数：以16位浮点数代替32位浮点数，使得内存使用量减半
    )
    
    # 创建聊天机器人
    chat_bot = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,  # 限制生成长度
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    # 准备对话历史
    conversation_history = [
        {"role": "user", "content": "你是谁？"},
    ]

    # 生成回答
    response = chat_bot(conversation_history)

    # 打印模型的回答
    print("Llama模型的回答:", response[0]['generated_text'])

except torch.cuda.OutOfMemoryError:
    print("GPU内存不足！请尝试以下解决方案：")
    print("1. 安装accelerate包来使用更高级的内存管理：")
    print("   pip install accelerate")
    print("2. 关闭其他占用GPU内存的程序")
    print("3. 使用更小的模型版本")
    print("4. 增加GPU内存")
    print("5. 使用CPU模式运行（会很慢）")
except Exception as e:
    print(f"发生错误: {str(e)}")
    print("\n如果你想使用更高级的内存管理功能，请安装accelerate：")
    print("pip install accelerate")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")