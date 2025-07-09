import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
import numpy as np
import json
from typing import List, Dict

def load_data(json_path: str) -> List[Dict]:
    """加载测试数据，假设格式为 [{ "input": ..., "label": 0/1, "group": 0/1 }, ...]"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

import re


def extract_default_status(text):
    """
    从文本中提取 Default/No Default，支持多种写法，返回1/0，无法识别返回None
    """
    if not isinstance(text, str):
        return None
    text = text.strip().lower()
    if 'default' in text:
        if 'no default' in text:
            return 0
        return 1
    elif '不违约' in text:
        return 0
    elif '违约' in text:
        return 1
    elif text == '1':
        return 1
    elif text == '0':
        return 0
    return None


def get_predictions(model, tokenizer, data: List[Dict], device="cuda"):
    preds, labels, groups = [], [], []
    model.eval()
    for item in data:
        text = item['input']
        # 真实标签从output字段提取，支持 Default/No Default/违约/不违约/0/1
        label = extract_default_status(item['output'])
        group_raw = item.get('group', 0)
        try:
            group = int(group_raw)
        except Exception:
            group = 0
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=48)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        pred = extract_default_status(decoded)
        # 若未能提取，默认为0
        if pred is None:
            pred = 0
        if label is None:
            # 若真实标签未能提取，跳过该样本
            continue
        preds.append(pred)
        labels.append(label)
        groups.append(group)
    return preds, labels, groups

def miss_value(y_true, y_pred):
    """Miss值 = FN / (TP + FN)"""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return fn / (tp + fn + 1e-8)

def disparate_impact(y_pred, groups):
    """差异影响 DI = P(Ŷ=1|G=1) / P(Ŷ=1|G=0)"""
    y_pred = np.array(y_pred)
    groups = np.array(groups)
    p1 = (y_pred[groups==1] == 1).mean()
    p0 = (y_pred[groups==0] == 1).mean()
    return p1 / (p0 + 1e-8)

def equal_opportunity_diff(y_true, y_pred, groups):
    """平等机会差异 EOD = TPR(G=1) − TPR(G=0)"""
    y_true = np.array(y_true); y_pred = np.array(y_pred); groups = np.array(groups)
    # True Positive Rate = TP / (TP + FN)
    mask1 = (groups==1) & (y_true==1)
    mask0 = (groups==0) & (y_true==1)
    tpr1 = ( (y_pred[mask1]==1).sum() ) / ( mask1.sum() + 1e-8 )
    tpr0 = ( (y_pred[mask0]==1).sum() ) / ( mask0.sum() + 1e-8 )
    return tpr1 - tpr0

def average_odds_diff(y_true, y_pred, groups):
    """平均赔率差异 AOD = 0.5 * [ (FPR1 − FPR0) + (TPR1 − TPR0) ]"""
    y_true = np.array(y_true); y_pred = np.array(y_pred); groups = np.array(groups)
    # TPR 部分
    mask1_pos = (groups==1) & (y_true==1)
    mask0_pos = (groups==0) & (y_true==1)
    tpr1 = ( (y_pred[mask1_pos]==1).sum() ) / ( mask1_pos.sum() + 1e-8 )
    tpr0 = ( (y_pred[mask0_pos]==1).sum() ) / ( mask0_pos.sum() + 1e-8 )
    # FPR 部分
    mask1_neg = (groups==1) & (y_true==0)
    mask0_neg = (groups==0) & (y_true==0)
    fpr1 = ( (y_pred[mask1_neg]==1).sum() ) / ( mask1_neg.sum() + 1e-8 )
    fpr0 = ( (y_pred[mask0_neg]==1).sum() ) / ( mask0_neg.sum() + 1e-8 )
    return 0.5 * ((fpr1 - fpr0) + (tpr1 - tpr0))

def main():

    # === 支持LoRA/PEFT分离式checkpoint加载 ===

    # 只用目录名，避免绝对路径被误判为repo_id
    base_model_path = "LLaMA-Factory/cModel"
    adapter_path = "LLaMA-Factory/saves/planA_lora/checkpoint-10000"
    data_path = "32final_testset_nl.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    print("加载数据...")
    data = load_data(data_path)

    print("开始推理与评估...")
    preds, labels, groups = get_predictions(model, tokenizer, data, device)

    # 计算常规指标
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds)
    mcc  = matthews_corrcoef(labels, preds)
    miss = miss_value(labels, preds)

    # 计算偏差指标
    di  = disparate_impact(preds, groups)
    eod = equal_opportunity_diff(labels, preds, groups)
    aod = average_odds_diff(labels, preds, groups)

    # 输出结果
    print(f"Accuracy:                      {acc:.4f}")
    print(f"F1 Score:                      {f1:.4f}")
    print(f"Matthews Correlation Coef.:    {mcc:.4f}")
    print(f"Miss Value (FN/(TP+FN)):      {miss:.4f}")
    print(f"Disparate Impact (DI):         {di:.4f}")
    print(f"Equal Opportunity Difference:  {eod:.4f}")
    print(f"Average Odds Difference:       {aod:.4f}")

if __name__ == "__main__":
    main()
