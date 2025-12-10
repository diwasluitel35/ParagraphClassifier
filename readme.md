# Paragraph Classifier

This repository contains my Automated Paragraph Classifier system built using **DeBERTa-v3-Base**.  
The project includes model training code, evaluation metrics, a demo web app, and a detailed technical report.

## Repository Structure

* `code/` - Full .ipynb file
* `dataset/` - Training and testing dataset
* `demo/` - Demo application
* `results/` - Evaluation table and visualizations
* `report.pdf` - Full technical report


## Model
The trained model is uploaded to Hugging Face:

Hugging Face Model: https://huggingface.co/diwasluitel/ParagraphClassifier

## Demo Usage
To run the demo:

1. Navigate to the `demo` folder.
2. Open a terminal in that folder.
3. Run the flask using command: python app.py
4. After it runs, open your browser and goto: http://127.0.0.1:5000

## UI Development

The front-end design was created with assistance from code generation tools to speed up development allowing the core project focus to remain exclusively on the back-end NLP model and evaluation process.

## Installation

Before running the demo, ensure you have necessary dependencies installed.

```powershell
pip install flask transformers torch sentencepiece protobuf