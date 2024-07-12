import torch
import json
import random
import tqdm

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def sample_paragraphs(squad_data, sample_data, num_paragraphs=3, questions_per_paragraph=3):
    predictions = {}
    if sample_data:
        sampled_paragraphs = random.sample(squad_data["data"], num_paragraphs)
    else:
        sampled_paragraphs = squad_data["data"]
    
    for article in tqdm.tqdm(sampled_paragraphs):
        if sample_data:
            paragraphs_to_process = random.sample(article["paragraphs"], 1)
        else:
            paragraphs_to_process = article["paragraphs"]
        
        for paragraph in paragraphs_to_process:
            text = paragraph["context"]
            if sample_data:
                qas = random.sample(paragraph["qas"], min(len(paragraph["qas"]), questions_per_paragraph))
            else:
                qas = paragraph["qas"]
            
            for qa in qas:
                question = qa["question"]
                try:
                    predictions[qa["id"]] = sliding_window_answer(question, text)
                except Exception as e:
                    print(f"Error processing question ID {qa['id']}: {str(e)}")
                    print(f"Text: {text}")
                    print(f"Question: {question}")
                    print(f"Error: {str(e)}")

    return predictions

# Adapted from https://colab.research.google.com/drive/1uSlWtJdZmLrI3FCNIlUHFxwAJiSu2J0-
def sliding_window_answer(question, answer_text, max_length=512, stride=128):
    '''
    Takes a `question` string and an `answer_text` string, processes them using
    a sliding window approach, and identifies the best answer from the text.
    '''
    question_tokens = tokenizer.encode(question, add_special_tokens=True)

    word_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    min_null_score = float('inf')  # large number
    best_start = 0
    best_end = 0

    for start in range(0, len(word_ids), stride):
        end = start + max_length - len(question_tokens) - 3  # [CLS], [SEP], [SEP]
        if end > len(word_ids):
            end = len(word_ids)

        chunk_ids = question_tokens + [tokenizer.sep_token_id] + word_ids[start:end] + [tokenizer.sep_token_id]

        input_ids = torch.tensor([chunk_ids]).to(device)
        segment_ids = torch.tensor([[0] * len(question_tokens) + [1] * (len(chunk_ids) - len(question_tokens))]).to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=segment_ids)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

        chunk_start_score, chunk_start_index = torch.max(start_scores, dim=1)
        chunk_end_score, chunk_end_index = torch.max(end_scores, dim=1)

        chunk_start_index = chunk_start_index.item() - len(question_tokens) - 1
        chunk_end_index = chunk_end_index.item() - len(question_tokens) - 1

        null_score = start_scores[0, 0] + end_scores[0, 0]  

        if null_score.item() < min_null_score:
            min_null_score = null_score.item()
            if chunk_start_index >= 0 and chunk_end_index >= chunk_start_index:
                best_start = start + chunk_start_index
                best_end = start + chunk_end_index

    final_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(word_ids[best_start:best_end+1]))
    return final_answer

file_path = 'dev-v2.0.json'

with open(file_path, 'r') as file:
    squad_data = json.load(file)

# Make Predictions
predictions_json = sample_paragraphs(squad_data, sample_data=False, num_paragraphs=3, questions_per_paragraph=3)
file_path = 'pred.json'
with open(file_path, 'w') as file:
    json.dump(predictions_json, file, indent=2)
