"""
Fine-tune LoRA do Llama 3.2 3B - script direto com transformers + peft
Sem Axolotl. Feito para GPU AMD (ROCm) com 8GB de VRAM.
"""

import torch

from transformers import (AutoModelForCausalLM,AutoTokenizer, TrainingArguments,Trainer,DataCollatorForLanguageModeling)

from peft import LoraConfig, get_peft_model

from datasets import load_dataset

# ============================================================
# CONFIGURAÇÃO 
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_PATH = "./data/meu_dataset.jsonl"   # formato: {"instruction": "...", "input": "...", "output": "..."}
OUTPUT_DIR = "./outputs/llama3.2-3b-lora"
MAX_LEN = 1024          # comece conservador com 8GB de VRAM; pode subir se sobrar memória
NUM_EPOCHS = 15  #Tinha 30 epochs aqui, irei reduzilas para 10
LEARNING_RATE = 1e-4 #Tinha 2e-4 #Notação ciêntifica aqui ou seja é: 0.0001
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4    # batch efetivo = BATCH_SIZE * GRAD_ACCUM_STEPS = 4

# ============================================================
# 1. Confirma que a GPU está disponível
# ============================================================
if not torch.cuda.is_available():
    raise RuntimeError("GPU não detectada! Confirme que o PyTorch ROCm está instalado corretamente.")
print(f"Usando GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 2. Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 3. Modelo em bf16 (sem quantização 4-bit - bitsandbytes tem suporte fraco em ROCm)
# ============================================================
print("Carregando modelo (isso baixa ~6.5GB na primeira vez)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False  # necessário junto com gradient checkpointing
print(f"Parâmetros totais do modelo: {model.num_parameters():,}")
# ============================================================
# 4. Configuração do LoRA
# ============================================================
lora_config = LoraConfig(
    r=8, #Irei ajustar o espaço para ele reescrever menos coisa no modelo.Tinha 16
    lora_alpha=16, #Tinha 32
    lora_dropout=0.01, #Tinha 0.05
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 5. Dataset - formato alpaca (instruction / input / output)
# ============================================================
print(f"Carregando dataset de {DATASET_PATH}...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")


def format_example(example):
    instruction = example["instruction"]
    input_text = example.get("input", "") or ""
    output = example["output"]
    if input_text:
        prompt = f"### Instrução:\n{instruction}\n\n### Entrada:\n{input_text}\n\n### Resposta:\n{output}"
    else:
        prompt = f"### Instrução:\n{instruction}\n\n### Resposta:\n{output}"
    return {"text": prompt + tokenizer.eos_token}


dataset = dataset.map(format_example)


def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_LEN)


tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# separa 5% pra validação
#split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
#train_dataset = split_dataset["train"]
#eval_dataset = split_dataset["test"]
train_dataset = tokenized_dataset
eval_dataset = None

#print(f"Exemplos de treino: {len(train_dataset)} | validação: {len(eval_dataset)}")
print(f"Exemplos de treino: {len(train_dataset)} | validação: nenhuma (dataset pequeno)")

# ============================================================
# 6. Configuração do treino
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    logging_steps=1,
    #eval_strategy="steps",
    #eval_steps=50,
    save_steps=100,
    save_total_limit=2,      # mantém só os 2 checkpoints mais recentes, economiza disco
    report_to="none",        # desativa wandb/tensorboard automático
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# ============================================================
# 7. Treinar
# ============================================================
print("Iniciando treino...")
trainer.train()

# ============================================================
# 8. Salvar o adapter LoRA (só os pesos ajustados, poucos MB)
# ============================================================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nTreino concluído! Adapter LoRA salvo em: {OUTPUT_DIR}")
