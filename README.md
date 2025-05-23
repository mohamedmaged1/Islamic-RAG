# Islamic Fiqh Rag

This projects aimes to retrieve islamic Fiqh answers from an islamic dataset and use Qwen-max to generate the answer

## Requirements 
- Python 3.10 or later 

### Install Python using Anaconda 
1) Download and install Anaconda from [here] (https://www.anaconda.com/download)
2) create a new environment using the following command:
```bash
$ conda create -n yalla python=10.0
```
3) Activate the environment:
```bash
$ conda activate yalla 
```

## Installation
### Install the reqired packages
```bash
$ pip install -r requirements.txt
```

### Setup the environment variables
```bash
cp .env.example .env
```

set your environment variable in the `.env` file. Like `OPENAI_API_KEY` value .

### For installing Arabic Language in tessdata 
On Windows :
1) Download and install Arabic from [here] (https://github.com/tesseract-ocr/tessdata/blob/main/ara.traineddata)
2) Place it into tessdata folder (e.g., C:\Program Files\Tesseract-OCR\tessdata)

### Creating RAG system 
#### Data
* I used a book name "ما لا يسع المسلم جهله " for retreiving.
* Then because the book in the form of PDF I converted it to a text.
* One problem I faced that to extract the text from the book I should used OCR that's why I used tessdata.
* Also, because the book contain Qura'an sentences with dicritics, sometimes when converting this using OCR I got wrong sentences.
* To solve this problem I wrote in the prompt to not include Qur'an answer in the generating.
* To convert your dataset, after running the code the all splits will be saved in order to load it again without rerunning the code
```bash
python handlingData.py
```



### LLM judge 
- 1) For more information about using LLM as a judge from [here] (https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
