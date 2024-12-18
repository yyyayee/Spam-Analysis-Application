import re
import json
import torch
import sentencepiece as spm
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from flask import Flask, request, jsonify, Response
from transformers import AutoModelForSequenceClassification

# Prepare stop words
stop_words = list(thai_stopwords())

def clean_thai_text(text):
    """Clean Thai text."""
    words = word_tokenize(text, engine='longest', keep_whitespace=False)
    words = [word for word in words if word not in stop_words]
    cleaned_words = [re.sub(r'[^ก-๙]', '', word) for word in words if word]
    return ' '.join(cleaned_words)

# Path to the SentencePiece model file
model_folder = r"C:\Users\Admin\Desktop\Project\Spam Project\wangchanberta-base-att-spm-uncased"

# Load SentencePiece tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(f"{model_folder}/sentencepiece.bpe.model")  # Use the SentencePiece model file

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_folder, num_labels=2)

def prepare_data_for_model(texts):
    """Prepare data for the model."""
    tokenized_inputs = [tokenizer.encode(text, out_type=int) for text in texts]  
    return {"input_ids": torch.tensor(tokenized_inputs)}

def predict(text):
    """Predict the label of a given text."""
    model.eval()
    inputs = prepare_data_for_model([text])
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()
    return prediction, confidence

# Flask app setup
app = Flask(__name__)

@app.route('/analyze_sms', methods=['POST'])
def analyze_sms():
    """API endpoint for analyzing SMS messages."""
    data = request.get_json()
    sms_text = data.get('text', '')

    if not sms_text:
        return jsonify({'error': 'กรุณาส่งข้อความ SMS มาในรูปแบบ JSON'}), 400

    # Clean and preprocess the text
    cleaned_message = clean_thai_text(sms_text)
    prediction, confidence = predict(cleaned_message)

    # Prepare result
    result = {
        'SMS นี้เป็น': f"{'Spam' if prediction else 'Ham'} {confidence * 100:.2f}%",
        'ข้อความ': sms_text
    }
    print(result)

    # Return JSON response with UTF-8 encoding
    response = Response(
        response=json.dumps(result, ensure_ascii=False),  # ensure_ascii=False ensures Thai is displayed correctly
        content_type="application/json; charset=utf-8",
        status=200
    )
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)