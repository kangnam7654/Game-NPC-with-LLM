class HuggingFaceWrapper:
    def __init__(self, model, tokenizer, system_prompt=None):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt or self.default_system_prompt()

    def chat(self, messages):
        if self.has_system_prompt(messages):
            messages = messages[1:]
        messages.insert(0, {"role": "system", "content": self.system_prompt})
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        output = self.model.generate(
            input_ids.to(self.model.device),
            max_new_tokens=512,
            do_sample=False,
        )
        generated_ids = output[0][input_ids.shape[1] :]
        decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return decoded_output

    def has_system_prompt(self, messages):
        to_check = messages[0]
        if to_check["role"] == "system":
            return True
        return False

    def default_system_prompt(self):
        return "You are a helpful assistant."
