"""
Chat interativo para testar o modelo com o adapter LoRA treinado.
Agora com: freio contra loops de repeticao + memoria da conversa.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "./outputs/llama3.2-3b-lora"
MAX_HISTORY_TURNS = 6  # quantos pares (usuario+modelo) manter no historico

print("Carregando modelo base (pode demorar um pouco)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("Aplicando o adapter LoRA treinado...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

print("\nPronto! Digite sua pergunta (ou 'sair' para encerrar, 'limpar' para reiniciar a conversa).\n")

# Historico da conversa - lista de {"role": ..., "content": ...}
messages = []

while True:
    user_input = input("\033[34mVocê:\033[0m")
    if user_input.strip().lower() in ("sair", "exit", "quit"):
        break
    if user_input.strip().lower() in ("limpar", "reset", "clear"):
        messages = []
        print("(histórico apagado)\n")
        continue

    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,     # penaliza o modelo por repetir palavras já ditas
            no_repeat_ngram_size=6,      # proíbe repetir a mesma sequência de 3 palavras | Era 3 aumentei para 6                         
            pad_token_id=tokenizer.eos_token_id,
            #output_scores=True,
            #return_dict_in_generate=True

        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\033[31mModelo:\033[0m {response}\n")

    # guarda a resposta no historico
    messages.append({"role": "assistant", "content": response})

    # limita o tamanho do historico (mantem só os ultimos N turnos, pra nao estourar o contexto)
    if len(messages) > MAX_HISTORY_TURNS * 2:
        messages = messages[-MAX_HISTORY_TURNS * 2:]



