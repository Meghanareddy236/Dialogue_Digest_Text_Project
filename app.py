from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)

# Check if GPU (CUDA) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
MODEL_DIR = "pegasus-samsum-model"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/tokenizer")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Get input text from the request
        input_text = request.json.get("text", "")
        if not input_text:
            return jsonify({"error": "No input text provided."}), 400
        
        # Tokenize the input text
        inputs = tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate the summary
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, length_penalty=0.8, max_length=128)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)