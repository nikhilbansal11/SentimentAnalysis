import os
import boto3
import torch
import torch.nn as nn
from torch.nn import functional as F
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# class SentimentClassifier(nn.Module):
#     def __init__(self, n_classes=5):
#         super(SentimentClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.drop = nn.Dropout(p=0.3)
#         self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

#     def forward(self, input_ids, attention_mask):
#         output = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         pooled_output = output.pooler_output
#         return self.fc(self.drop(pooled_output))

# def download_model():
#     """Download model from S3 bucket"""
#     try:
#         s3 = boto3.client('s3')
#         s3.download_file('iitmodel', 'model.tar.gz', '/tmp/model.tar.gz')
#         os.system('tar -xzf /tmp/model.tar.gz -C /tmp/')
#         logger.info("Model downloaded and extracted successfully")
#     except Exception as e:
#         logger.error(f"Error downloading model: {str(e)}")
#         raise

# def load_model():
#     """Load the model and tokenizer"""
#     try:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = SentimentClassifier()
#         model.load_state_dict(torch.load('/tmp/best_model.pt', map_location=device))
#         model.to(device)
#         model.eval()
#         return model, tokenizer, device
#     except Exception as e:
#         logger.error(f"Error loading model: {str(e)}")
#         raise

# def predict_sentiment(text, model, tokenizer, device):
#     """Process text and return sentiment prediction"""
#     encoding = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=128,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )

#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)

#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask)
#         _, prediction = torch.max(outputs, dim=1)
#         probabilities = F.softmax(outputs, dim=1)

#     return prediction.item() + 1, probabilities.cpu().numpy()[0].tolist()

# Download and load model at startup
# download_model()
# model, tokenizer, device = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        
        # Get prediction
        # rating, probabilities = predict_sentiment(text, model, tokenizer, device)
        
        # Create response
        # sentiment_map = {
        #     1: "Very Negative",
        #     2: "Negative",
        #     3: "Neutral",
        #     4: "Positive",
        #     5: "Very Positive"
        # }
        
        # response = {
        #     'rating': rating,
        #     'sentiment': sentiment_map[rating],
        #     'confidence_scores': {
        #         f'rating_{i+1}': prob 
        #         for i, prob in enumerate(probabilities)
        #     }
        # }

        response = {
            "confidence_scores": {
                "rating_1": 0.001673496444709599,
                "rating_2": 0.0008952409843914211,
                "rating_3": 0.001788665191270411,
                "rating_4": 0.04023696109652519,
                "rating_5": 0.9554056525230408
            },
            "rating": 5,
            "sentiment": "Very Positive"
        }
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
