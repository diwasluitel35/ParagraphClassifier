import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

app = Flask(__name__)

class AttentionPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state, attention_mask):
        attention_scores = self.attention(last_hidden_state)
        mask = attention_mask.unsqueeze(-1)
        attention_scores[mask == 0] = -1e4
        attention_weights = torch.softmax(attention_scores, dim=1)
        pooled_output = torch.sum(attention_weights * last_hidden_state, dim=1)
        return pooled_output

class ParagraphClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        hidden = self.transformer.config.hidden_size
        self.pooler = AttentionPooler(hidden_size=hidden)
        self.classifier = nn.Linear(hidden, 7)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        pooled = self.pooler(last_hidden_state=last_hidden, attention_mask=attention_mask)
        logits = self.classifier(pooled)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

model_path = 'diwasluitel/ParagraphClassifier'
base_model = 'microsoft/deberta-v3-base'

def load_model():
    global tokenizer, model, label_encoder
    try:
        print(f"Loading model: {model_path}...")
        path = hf_hub_download(repo_id=model_path, filename="label_encoder.pkl")
        label_encoder = joblib.load(path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = ParagraphClassifier(model=base_model)
        model_weights_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        state_dict = load_file(model_weights_path)
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model (Demo Mode Active): {e}")
        tokenizer = None
        model = None

load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Please enter a paragraph.'}), 400

    if model and tokenizer:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            inputs = f"{text}"
            tokens = tokenizer(inputs, padding=False, truncation=True, max_length=256, return_tensors='pt')
            tokens = {k: v.to(device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = model(**tokens)
                logits = outputs['logits']
                pred_id = np.argmax(logits.cpu().numpy(), axis=-1)
                pred_label = label_encoder.inverse_transform(pred_id)[0]

            return jsonify({"result": pred_label})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({"error": "Model not loaded."}), 500

if __name__ == '__main__':
    app.run(debug=True)