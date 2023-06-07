from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import re
import sentiments
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import TextClassificationPipeline
import KEY # mykey.py에 openai api key를 입력 후, (import KEY => import mykey)로 변경

app = Flask(__name__)
CORS(app)

# BERT_PATH: hugging-face에 올라가 있는 모델 (https://huggingface.co/burningfalls/my-fine-tuned-bert)
BERT_PATH = "burningfalls/my-fine-tuned-bert"
GPT_NAME = "gpt-3.5-turbo"
# GPT_OPTION: GPT 커스텀
GPT_OPTION = "친구, 일상대화, 반말, ';;' 뒤에 있는 감정을 답변 문장에 반영"


# text_classifier: BERT 파이프라인
global text_classifier
# 일정 길이(10)의 messages를 관리할 queue
global queue
queue = list()

# BERT model을 사용할 수 있게 만들어주는 함수
def load_bert():
    global text_classifier

    # 감성 분류를 위한 tokenizer와 model 로드
    loaded_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    loaded_model = TFAutoModelForSequenceClassification.from_pretrained(BERT_PATH)

    # 텍스트 분류용 파이프라인 생성
    text_classifier = TextClassificationPipeline(
        tokenizer=loaded_tokenizer,
        model=loaded_model,
        framework='tf',
        top_k=5
    )


# BERT
# input: question => BERT => output: feel_list
def predict_sentiment(question):
    global text_classifier

    # 질문에 대한 감성 예측 + 인덱스 추출
    result_list = text_classifier(question)[0]
    # sentiments.py 파일에서 해당 감정 인덱스에 대응하는 감정 레이블 매칭
    feels = []
    for result in result_list:
        feel_idx = int(re.sub(r'[^0-9]', '', result['label']))
        feel = sentiments.Feel[feel_idx]["label"]
        feels.append(feel)

    return feels


# GPT
# input: question+feel => GPT => output: answer
def generate_answer(question, feel):
    global queue

    # queue의 길이가 10이면
    # 제일 오래된(왼쪽) 메시지(user & assistant) 두 개를 없앰
    if len(queue) == 10:
        for i in range(2):
            queue.pop(0)

    # {질문+감정}을 queue에 추가 (role: user)        
    queue.append(question[:(min(len(question), 128))] + ' ;; ' + feel)

    # queue를 기반으로, 최근 질문&답변을 이어붙여 GPT의 input으로 넣을 message 생성
    messages = [{"role": "system", "content": GPT_OPTION}]
    for i in range(0, len(queue)):
        if i % 2 == 0:
            messages.append({"role": "user", "content": queue[i]})
        elif i % 2 == 1:
            messages.append({"role": "assistant", "content": queue[i]})

    # OpenAI Chat API를 사용하여 답변 생성
    result = openai.ChatCompletion.create(
        model=GPT_NAME,
        messages=messages
    )   
    answer = result.choices[0].message.content

    # {답변}을 queue에 추가 (role: assistant)
    queue.append(answer[:(min(len(question), 128))])
    
    return answer


@app.route('/predict', methods=['POST'])
def predict():
    # POST 요청에서 질문(question) 가져오기
    question = request.get_json()['text']
    # input: question => BERT => output: feel_list
    feel_list = predict_sentiment(question)
    # input: question+feel => GPT => output: answer
    answer = generate_answer(question, feel_list[0])

    return jsonify({'feel_list': feel_list, 'result': answer})


@app.route('/predict-again', methods=['POST'])
def predict_again():
    # POST 요청에서 질문(question)과 감정(feel) 가져오기
    question = request.get_json()['text']
    feel = request.get_json()['feel']

    # input: question+feel => GPT => output: answer
    answer = generate_answer(question, feel)

    return jsonify({'feel_list': '', 'result': answer})


if __name__ == '__main__':
    # BERT model 로드
    load_bert()
    # Flask 애플리케이션 실행
    app.run(host='0.0.0.0')