# Fiducia

**Fiducia** — A CALM‑derived credit scoring model with enhanced compliance alignment.

> **Focused on Credit Scoring**: This project is dedicated to optimizing compliance and performance for credit scoring tasks using a fine‑tuned LLaMA2-Chat-HF model.
>
> GitHub Repository: [https://github.com/Jackksonns/Fiducia](https://github.com/Jackksonns/Fiducia)

This repository implements **Fiducia**, a credit scoring framework built upon the CALM model (as described in *Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models*), with additional compliance alignment strategies tailored for financial risk assessment.

---

## 🚀 Key Contributions

* **Compliance-Driven Design**: We integrate regulatory alignment techniques into the original CALM architecture, enhancing interpretability and auditability in lending scenarios.
* **Multi-Source Data Integration**: The framework consolidates Australian, German, Lending Club, and Home Credit datasets spanning Q1 2007 to Q3 2020 to ensure robust and generalizable performance.
* **End-to-End Pipeline**: Includes modules for data conversion, fine‑tuning, compliance evaluation, and stratified sampling for testing.
* **Reproducibility**: Comprehensive configuration files, scripts, and parameter documentation facilitate replication and customization.

---

## 📚 Background and Citation

The CALM model is introduced in:

> **Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models**
> The authors demonstrate the generalization capabilities of LLMs across heterogeneous credit datasets.

Please cite the original work as follows:

```bibtex
@inproceedings{calm2025,
  title={Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models},
  author={Author et al.},
  booktitle={Proceedings of XXX},
  year={2025}
}
```
---

## 🔧 Fine-Tuning Methodology

Fiducia leverages Llama-Factory’s visual fine‑tuning interface to streamline the adaptation of LLaMA2-Chat-HF for credit scoring tasks. This approach enables:

#Intuitive Hyperparameter Adjustment#: Researchers can interactively modify learning rates, LoRA rank, and regularization parameters via an integrated dashboard, promoting efficient experimentation.

#Real‑Time Training Visualization#: Loss curves, metric trajectories, and attention pattern heatmaps are rendered live, facilitating rapid diagnostics and early stopping decisions.

#Seamless LoRA Integration#: The platform automates low‑rank adaptation injection and merging workflows, ensuring consistent reproducibility and compliance with Meta’s open‑source requirements.

---

## 📂 Datasets

### Original CALM Data Sources

1. [Statlog Australian Credit Approval (UCI)](http://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval)
2. [Statlog German Credit Data (UCI)](http://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
3. (To Be Updated) [Lending Club 2007](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

### Additional Data for This Project

1. [Lending Club Clean 2007–Q3 2020 (Kaggle)](https://www.kaggle.com/datasets/marcusos/lending-club-clean)
2. [Accepted Loans 2007–Q4 2018 Lending Club (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
3. [Home Credit Default Risk (Kaggle Competition)](https://www.kaggle.com/competitions/home-credit-default-risk/data)

**Test Set Construction**: A stratified sample is drawn across the `loan_status` field from the three sources to form a mixed test set covering data from 2007–Q3 2020 (Lending Club), pre‑Q4 2018 accepted loans, and Home Credit default risk records.

---

## ⚙️ Setup and Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Jackksonns/Fiducia.git
   cd Fiducia
   ```
2. **Create and activate a Python environment** (recommended: `venv` or `conda`)

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🗂️ Project Structure

```text
fiducia/
├── dataset/               # Scripts for dataset format conversion
├── test/                  # Test scripts and examples
├── inference_example.py   # Example inference script
├── lora_finetune.py       # LoRA fine‑tuning entry point
├── merge_lora_to_llama.py # Merge LoRA weights into LLaMA2
├── merge_testsets.py      # Stratified sampling and test set merging
├── requirements.txt       # Python dependencies
├── README.md              # This document
└── LICENSE                # License file
```

> **Note**: Per Meta’s open‑source policy, LLaMA2 model weights are **not** included in this repository. The `dataset/` folder contains only data conversion scripts; raw datasets must be obtained separately.

---

## 🔨 Quick Start

The following commands illustrate a typical end‑to‑end workflow. Adjust script names as needed based on the repository contents:

1. **Data Preparation**

   ```bash
   # Convert and clean raw data
   python dataset/your_data_prep_script.py
   ```
2. **Model Fine‑Tuning**

   ```bash
   python lora_finetune.py --config configs/train.yaml
   ```
3. **Evaluation and Inference**

   ```bash
   python merge_testsets.py --config configs/data.yaml
   python inference_example.py --model-path path/to/fiducia
   ```

---

## 📈 Results and Evaluation

We evaluate Fiducia on both predictive performance and fairness metrics to ensure a balanced improvement:

Performance Metrics (e.g., Accuracy, F1‑Score, MCC, Recall)

Fairness Metrics (e.g., Equal Opportunity Difference, Average Odds Difference)

Across our mixed test set, Fiducia consistently outperforms the original CALM baseline on both fronts, demonstrating enhanced predictive accuracy while reducing fairness disparities.

---

## 📝 Contribution and License

Contributions via Pull Requests are welcome. Please review \[CONTRIBUTING.md] before submitting.

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

© 2025 Fiducia Alignment Team | Maintainer: Jackson KK ([2963087383@qq.com](mailto:2963087383@qq.com))
