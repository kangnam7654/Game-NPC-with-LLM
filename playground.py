import re

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="bfloat16", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# choose your prompt
prompt = "Explain how wonderful you are"
prompt = "Explica lo increíble que eres"
prompt = "너가 얼마나 대단한지 설명해 봐"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=False,  # Use greedy decoding
)

# --- 파싱 로직 ---

# 1. 생성된 토큰만 디코딩합니다.
generated_ids = output[0][input_ids.shape[1] :]
decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=False)

print("--- 전체 생성 결과 ---")
print(decoded_output)

# 2. <think> 블록을 파싱합니다.
think_content = ""
think_match = re.search(r"<think>(.*?)</think>", decoded_output, re.DOTALL)
if think_match:
    think_content = think_match.group(1).strip()

# 3. <think> 블록을 제거하고 실제 답변만 추출합니다.
response_text = re.sub(
    r"<think>.*?</think>", "", decoded_output, flags=re.DOTALL
).strip()

# 4. [|assistant|] 이후의 텍스트만 가져옵니다. (만약 있다면)
if response_text.startswith("[|assistant|]"):
    response_text = response_text.split("[|assistant|]", 1)[1].strip()

print("\n--- 파싱된 결과 ---")
print(f"생각: {think_content if think_content else '없음'}")
print(f"답변: {response_text}")
