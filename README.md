# Memes Generator Using Machine Learning

A Flask web app that suggests meme captions from a topic or situation. It ranks sample captions using TF-IDF text features and cosine similarity. No API key is needed for this demo.

## Features
- Enter a topic or situation
- Rank matching sample caption ideas
- Simple web interface

## Technologies
Python, Flask, scikit-learn, HTML, CSS

## Run
```bash
python -m venv .venv
```
Activate the environment, then:
```bash
pip install -r requirements.txt
python app.py
```
Visit http://127.0.0.1:5000.

## Note
This starter recommends text captions; it does not generate meme images or connect to Reddit by default. Describe only features you have actually implemented and tested.
