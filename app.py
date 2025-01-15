import streamlit as st
import torch
from transformers import BertTokenizer
import torch.nn as nn
from torch.nn import functional as F

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super(SentimentClassifier, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = output.pooler_output
        return self.fc(self.drop(pooled_output))

@st.cache_resource
def load_model():
    """Load the model and tokenizer only once and cache them"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SentimentClassifier()
    model.load_state_dict(torch.load(r'C:\Users\Admin\Downloads\iikgp_sentiment_analysis\best_model.pt', map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device):
    """Process text and return sentiment prediction"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs, dim=1)
        probabilities = F.softmax(outputs, dim=1)

    return prediction.item() + 1, probabilities.cpu().numpy()[0]

def main():
    st.set_page_config(
        page_title="Review Sentiment Analyzer",
        page_icon="📊",
        layout="centered"
    )

    st.title("📊 Review Sentiment Analyzer")
    st.write("Enter your review text below to analyze its sentiment.")

    # Load model (will be cached)
    try:
        model, tokenizer, device = load_model()
        model_load_success = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_load_success = False

    # Text input
    review_text = st.text_area(
        "Review Text",
        height=100,
        placeholder="Type or paste your review here..."
    )

    # Add analyze button
    if st.button("Analyze Sentiment"):
        if not model_load_success:
            st.error("Cannot analyze: Model failed to load")
            return

        if not review_text.strip():
            st.warning("Please enter some text to analyze")
            return

        with st.spinner("Analyzing..."):
            try:
                rating, probabilities = predict_sentiment(review_text, model, tokenizer, device)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Show predicted rating
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Predicted Rating", f"{rating}/5")
                
                # Show confidence scores
                with col2:
                    st.write("Confidence Scores:")
                    for i, prob in enumerate(probabilities):
                        st.progress(float(prob))
                        st.write(f"Rating {i+1}: {prob:.2%}")

                # Sentiment interpretation
                sentiment_map = {
                    1: "Very Negative 😡",
                    2: "Negative ☹️",
                    3: "Neutral 😐",
                    4: "Positive 😊",
                    5: "Very Positive 😄"
                }
                st.subheader(f"Overall Sentiment: {sentiment_map[rating]}")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
