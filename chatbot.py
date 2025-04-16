import sys
from ctransformers import AutoModelForCausalLM

sys.stdout.reconfigure(encoding='utf-8')

try:
    user_input = sys.argv[1]
except IndexError:
    print("No input received.")
    exit()

try:
    print("[INFO] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        ".",
        model_file="openchat-3.5.Q4_K_M.gguf",
        model_type="llama"
    )
    print("[INFO] Model loaded.", flush=True)
    
    prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
    print("[INFO] Generating response...", flush=True)
    response = model(prompt, max_new_tokens=16, stop=["<|user|>", "<|assistant|>", "<|system|>"])
    print("[INFO] Done.")
    print(response.strip())

except Exception as e:
    print("Sorry, I couldn't process that right now.")
