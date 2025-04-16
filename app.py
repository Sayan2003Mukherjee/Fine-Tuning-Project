from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

@app.route('/generate', methods=['POST'])
def generate_code():
    prompt = request.json.get('prompt', '')
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=256, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"generated_code": generated_code})

if __name__ == '__main__':
    app.run(debug=True)
