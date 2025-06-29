# Fine-Tuning Mistral-7B on Instruction-Based Dataset

This project explores the fine-tuning of a large open-source instruction-tuned model using a symbolic instruction dataset. The goal is to enhance the model's ability to follow instructions accurately and improve its performance on downstream tasks.

---

## 1. 🧠 Model Selection

- **Model**: `mistralai/Mistral-7B-Instruct-v0.1`
- An efficient and powerful open-source model designed for instruction-following tasks.

---

## 2. 📊 Dataset Preparation

- **Dataset**: `sail/symbolic-instruction-tuning`
- Each instance is formatted as a chat-style instruction with a consistent system prompt:
  > *"You are a helpful assistant. Please follow the instruction and provide a helpful response."*
- A custom chat template was used to wrap instructions before training.

---

## 3. ⚙️ Training Configuration

- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Frameworks**: Hugging Face Transformers, PEFT, TRL
- **Trainer**: `SFTTrainer` from TRL
- **Epochs**: 1  
- **Batch size**: 4  
- **Gradient Accumulation**: 2  
- **Learning Rate**: `2e-4`  
- **Optimizer**: `paged_adamw_32bit`  
- **Precision**: `fp16`

**LoRA Configuration**:
- `r = 8`
- `alpha = 32`
- `dropout = 0.05`
- `target_modules = [q_proj, k_proj, v_proj, o_proj]`

---

## 4. 🖥️ Compute Used

- **Platform**: Kaggle Notebooks
- **GPU**: NVIDIA P100
- **RAM**: 25 GB

---

## 5. 📈 Evaluation

Model was evaluated before and after fine-tuning using a fixed set of prompts. The following metrics were used:

| Metric       | Before FT | After FT |
|--------------|-----------|----------|
| F1-Score     |   –       |   ✓      |
| ROUGE-L      |   –       |   ✓      |
| BERTScore    |   –       |   ✓      |

> *(Detailed results available inside the notebook `better.ipynb`)*

---

## 📁 Files

- `better.ipynb` – Full fine-tuning code and evaluation
- `report.docx` – Formal write-up and documentation
