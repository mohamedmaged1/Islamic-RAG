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