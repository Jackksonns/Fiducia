# 基础库
import os  # 用于处理文件和目录路径
import torch  # PyTorch深度学习框架，用于张量计算和GPU加速
import argparse  # 用于解析命令行参数

# Hugging Face Transformers库相关组件
from transformers import (
    AutoModelForCausalLM,  # 自动加载因果语言模型（如Llama）的类
    AutoTokenizer,  # 自动加载对应模型的分词器
    TrainingArguments,  # 训练参数配置类，用于设置训练相关的超参数
    Trainer,  # 训练器类，封装了训练循环和评估逻辑
    DataCollatorForLanguageModeling,  # 数据整理器，用于将多个样本整理成批次
    BitsAndBytesConfig  # 用于配置模型量化参数
)

# PEFT (Parameter-Efficient Fine-Tuning) 相关组件
from peft import (
    prepare_model_for_kbit_training,  # 准备模型进行低比特训练
    LoraConfig,  # LoRA配置类，用于设置LoRA的参数
    get_peft_model,  # 将普通模型转换为PEFT模型
    PeftModel  # PEFT模型基类
)

# 数据集处理
from datasets import load_dataset  # 用于加载和处理数据集的工具

def parse_args(): #参数解析器，设置命令行可输入参数
    # 创建一个参数解析器，description是帮助信息
    parser = argparse.ArgumentParser(description='LoRA Fine-tune Llama model')
    
    # 添加命令行参数
    parser.add_argument(
        '--data_path',           # 参数名
        type=str,                # 参数类型是字符串
        required=True,           # 这个参数是必需的
        help='Path to training data'  # 帮助信息
    )
    
    parser.add_argument(
        '--output_dir',          # 输出目录参数
        type=str,                # 字符串类型
        default='models/lora_finetuned',  # 默认值
        help='Output directory for model'  # 帮助信息
    )
    
    parser.add_argument(
        '--num_epochs',          # 训练轮数
        type=int,                # 整数类型
        default=3,               # 默认训练3轮
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',          # 批次大小
        type=int,                # 整数类型
        default=4,               # 默认每批4个样本
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning_rate',       # 学习率
        type=float,              # 浮点数类型
        default=2e-4,            # 默认学习率0.0002
        help='Learning rate'
    )
    
    # 解析命令行参数并返回
    return parser.parse_args()

def prepare_dataset(data_path, tokenizer):
    """准备数据集"""
    # 加载数据集
    dataset = load_dataset('json', data_files=data_path)
    
    def tokenize_function(examples):
        """对文本进行分词"""
        # 构建对话格式
        conversations = []
        for text in examples['text']:
            # 这里假设每个样本都是单轮对话
            # 实际使用时可能需要根据你的数据格式调整
            conversation = f"Human: {text}\n\nAssistant: "
            conversations.append(conversation)
        
        # 对文本进行编码
        return tokenizer(
            conversations,
            padding=True,  # 启用padding
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    # 对数据集进行分词
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model_name = "./model"  # 使用本地模型路径
    
    # 配置8位量化
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    # 加载tokenizer并设置padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # 在右侧进行padding
    
    # 确保模型知道padding token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,  # 使用半精度浮点数：以16位浮点数代替32位浮点数，使得内存使用量减半
        pad_token_id=tokenizer.pad_token_id  # 设置模型的padding token id
    )
    
    # 准备模型进行LoRA训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=8,  # LoRA的秩
        lora_alpha=32,  # LoRA的alpha参数
        target_modules=["q_proj", "v_proj"],  # 需要训练的模块
        lora_dropout=0.05,  # Dropout概率，防止过拟合
        bias="none",
        task_type="CAUSAL_LM"  # 任务类型：因果语言模型
    )
    
    # 将模型转换为PEFT模型
    model = get_peft_model(model, lora_config)
    
    # 准备数据集
    print("Preparing dataset...")
    dataset = prepare_dataset(args.data_path, tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=True,  # 使用混合精度训练
        gradient_accumulation_steps=4,  # 梯度累积步数
    )
    
    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main()