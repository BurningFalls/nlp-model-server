from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import re
import KEY # mykey.py에 openai api key를 입력 후, (import KEY => import mykey)로 변경
import sentiments
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import TextClassificationPipeline

app = Flask(__name__)
CORS(app)

BERT_PATH = "burningfalls/my-fine-tuned-bert"
GPT_NAME = "gpt-3.5-turbo"

global text_classifier
global messages
messages = []

# BERT 모델 load
def load_model():
    global text_classifier

    loaded_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    loaded_model = TFAutoModelForSequenceClassification.from_pretrained(BERT_PATH)

    text_classifier = TextClassificationPipeline(
        tokenizer=loaded_tokenizer,
        model=loaded_model,
        framework='tf',
        top_k=1
    )


# 서버에서 받은 텍스트 데이터를 BERT 모델로 예측하는 함수
def predict_sentiment(text):
    global text_classifier

    pred = text_classifier(text)[0]
    predicted_label = int(re.sub(r'[^0-9]', '', pred[0]['label']))

    return predicted_label


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    content = data['text']
    feel_idx = predict_sentiment(content)  # bert출력

    # gpt+bert
    feel = sentiments.Feel[feel_idx]["label"]
    messages.append({"role": "user", "content": content + feel})

    # 감정 문장 생성
    completion = openai.ChatCompletion.create(
        model=GPT_NAME,
        messages=messages
    )
    chat_response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": chat_response})

    return jsonify({'result': chat_response})


if __name__ == '__main__':
    load_model()
    messages.append({"role": "system", "content": "친구, 일상대화, 반말"})
    app.run()