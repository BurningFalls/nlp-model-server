# NLP Model Server
**이 프로젝트는 자연어 처리(NLP) 모델을 활용한 웹 서비스를 제공한다.** `Flask` framework를 사용하여 구축되었으며, `fine-tuned BERT`와 `GPT-3.5 Turbo` 모델을 통해 감정 분류 및 자동 응답 기능을 제공한다. 이를 통해 사용자는 자연어 질문을 입력하고, 해당 질문에 대한 감정이 반영된 AI 기반의 응답을 받을 수 있다.

<br>

## 1. 기능
* **감정 분류**: 입력된 질문에 대한 감정 분류를 수행한다. `fine-tuned BERT`를 사용하여 감정 분류 모델을 구현하였으며, pretrained model을 활용한다. 감정 레이블은 사전에 정의된 감정 index를 기반으로 하여 사용자에게 전달된다.
* **자동 응답**: 입력된 질문과 감정 정보를 기반으로, `GPT-3.5 Turbo model`을 활용하여 자동 응답을 생성한다. AI 기반의 대화 모델을 사용하여 자연스러운 대화를 제공한다.

<br>

## 2. 파일 구조
```
app
 ├── app.py
 ├── mykey.py
 ├── sentiments.py
.gitignore
README.md
requirements.txt
```
* `app`: 애플리케이션 폴더
  * `app.py`: Flask application의 메인 파일. 웹 서비스의 endpoint와 핵심 기능이 구현되어 있다.
  * `mykey.py`: OpenAI API key를 저장하는 파일. 필요한 경우, OpenAI API key를 이 파일에 입력해야 한다.
  * `sentiments.py`: 감정 분류에 사용되는 감정 label과 index를 정의한 파일.
* `.gitignore`: Git으로 관리하지 않을 file 및 directory를 명시한 파일.
* `README.md`: 현재 문서
* `requirements.txt`: 프로젝트에 필요한 Python package들과 version 정보를 명시한 파일.

<br>

## 3. 사용 방법
1. `mykey.py` file에서 OpenAI API key를 입력한다.
2. 필요한 Python 패키지들을 설치하기 위해 다음 명령을 실행한다.
```
pip install -r requirements.txt
```
3. Flask application을 실행한다.
```
python app/app.py
```
4. Web browser에서 `http://localhost:5000` 으로 접속하여 NLP model service를 이용할 수 있다.
