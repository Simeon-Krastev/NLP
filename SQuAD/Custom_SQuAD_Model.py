import json
import random
import os
from tqdm import tqdm

from transformers import BertTokenizerFast, BertModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def load_data(filename, num_samples=None):
    with open(filename, 'r') as file:
        data = json.load(file)['data']
    if num_samples is not None and num_samples < len(data):
        data = random.sample(data, num_samples)
    print("Data loaded successfully.")
    return prepare_articles(data)

def split_articles(data, train_split=0.8, val_split=0.1):
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

def prepare_articles(data):
    articles = []
    for article in tqdm(data, desc="Preparing articles"):
        article_dict = {
            'title': article['title'],
            'paragraphs': []
        }
        for paragraph in article['paragraphs']:
            para_dict = {
                'context': paragraph['context'],
                'qas': []
            }
            for qa in paragraph['qas']:
                qa_dict = {
                    'question': qa['question'],
                    'id': qa['id'],
                    'is_impossible': qa['is_impossible'],
                    'answers': qa['answers'] if not qa['is_impossible'] else []
                }
                para_dict['qas'].append(qa_dict)
            article_dict['paragraphs'].append(para_dict)
        articles.append(article_dict)
    print("Articles prepared successfully.")
    return articles

def save_test_data(test_data, filename='test_data.json'):
    squad_format = {'version': 'v2.0', 'data': test_data}
    with open(filename, 'w') as file:
        json.dump(squad_format, file, indent=4)
    print(f"Test data saved to {filename}.")
    
class QuestionAnsweringDataset(Dataset):
    def __init__(self, articles, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        for article in articles:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    self.data.append({
                        'context': context,
                        'question': qa['question'],
                        'id': qa['id'],
                        'is_impossible': qa['is_impossible'],
                        'answers': qa['answers']
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question, context = item['question'], item['context']
        is_impossible = item['is_impossible']
        question_id = item['id']

        encoded = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation='only_second',
            return_attention_mask=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        offset_mapping = encoded['offset_mapping'].squeeze(0)

        start_positions = end_positions = torch.tensor(0, dtype=torch.long)

        if not is_impossible and item['answers']:
            answer = item['answers'][0]
            start_char = answer['answer_start']
            end_char = start_char + len(answer['text']) - 1

            start_token_idx = [i for i, offset in enumerate(offset_mapping) if offset[0] == start_char]
            end_token_idx = [i for i, offset in enumerate(offset_mapping) if offset[1] == end_char + 1]

            if start_token_idx:
                start_positions = torch.tensor(start_token_idx[0], dtype=torch.long)
            if end_token_idx:
                end_positions = torch.tensor(end_token_idx[0], dtype=torch.long)

        max_index = input_ids.size(0) - 1
        start_positions = torch.clamp(start_positions, 0, max_index)
        end_positions = torch.clamp(end_positions, 0, max_index)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'is_impossible': torch.tensor(is_impossible, dtype=torch.float),
            'question_id': question_id
        }

class UQuestionAnsweringModel(nn.Module):
    def __init__(self, bert_model='bert-base-uncased'):
        super(UQuestionAnsweringModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.start_logits = nn.Linear(self.bert.config.hidden_size, 1)
        self.end_logits = nn.Linear(self.bert.config.hidden_size, 1)
        self.unanswerable_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        start_logits = self.start_logits(sequence_output).squeeze(-1)
        end_logits = self.end_logits(sequence_output).squeeze(-1)
        unanswerable_logits = self.unanswerable_classifier(sequence_output[:, 0, :]).squeeze(-1) 

        return start_logits, end_logits, unanswerable_logits

def compute_loss(start_logits, end_logits, unanswerable_logits, start_positions, end_positions, is_impossible):
    ce_loss = CrossEntropyLoss(ignore_index=-1)
    start_loss = ce_loss(start_logits, start_positions.clamp(0, start_logits.size(1) - 1))
    end_loss = ce_loss(end_logits, end_positions.clamp(0, end_logits.size(1) - 1))
    
    bce_loss = BCEWithLogitsLoss()
    unanswerable_loss = bce_loss(unanswerable_logits, is_impossible)

    return start_loss + end_loss + unanswerable_loss

def train(model, data_loader, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                is_impossible = batch['is_impossible'].to(device)

                optimizer.zero_grad()
                start_logits, end_logits, unanswerable_logits = model(input_ids, attention_mask)
                loss = compute_loss(start_logits, end_logits, unanswerable_logits, start_positions, end_positions, is_impossible)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error during training: {str(e)}")
                continue
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(data_loader)}")

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            is_impossible = batch['is_impossible'].to(device)

            start_logits, end_logits, unanswerable_logits = model(input_ids, attention_mask)
            loss = compute_loss(start_logits, end_logits, unanswerable_logits, start_positions, end_positions, is_impossible)
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f"Average Loss: {average_loss}")
    return average_loss

def generate_predictions(model, data_loader, tokenizer, threshold=0.5):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating Predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            start_logits, end_logits, unanswerable_logits = model(input_ids, attention_mask)
            unanswerable_probs = torch.sigmoid(unanswerable_logits)

            for idx in range(input_ids.size(0)):
                question_id = batch['question_id'][idx]
                if unanswerable_probs[idx] > threshold:
                    predictions[question_id] = ""
                else:
                    start_idx = torch.argmax(start_logits[idx]).item()
                    end_idx = torch.argmax(end_logits[idx]).item()
                    if start_idx > end_idx:
                        predictions[question_id] = ""
                    else:
                        tokens = tokenizer.convert_ids_to_tokens(input_ids[idx][start_idx:end_idx + 1], skip_special_tokens=True)
                        answer = tokenizer.convert_tokens_to_string(tokens)
                        predictions[question_id] = answer

    with open('predictions.json', 'w') as f:
        json.dump(predictions, f, indent=4)

    print("Predictions have been saved to 'predictions.json'")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

filename = 'dev-v2.0.json'

print("Loading Data...")
articles = load_data(filename, 15)

print("Splitting into Training/Validation/Testing Sets...")
train_data, val_data, test_data = split_articles(articles)

print("Saving Test Data for Evaluation...")
save_test_data(test_data)

print("Processing Data...")
train_dataset = QuestionAnsweringDataset(train_data, tokenizer)
val_dataset = QuestionAnsweringDataset(val_data, tokenizer)
test_dataset = QuestionAnsweringDataset(test_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

print("Setting up the Architecture...")
model = UQuestionAnsweringModel()
model = UQuestionAnsweringModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

print("Training the Model...")
train(model, train_loader, optimizer, num_epochs=8)

# Save model weights to re-use training for model predictions:
model_save_path = 'uqa_model_weights.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}.")

# If there are weights to load:
# model.load_state_dict(torch.load('uqa_model_weights.pth'))
# print("Model weights loaded and model is ready for inference.")

print("Evaluating the Model...")
evaluate(model, val_loader)

print("Generating Predictions on the Test Set...")
generate_predictions(model, test_loader, tokenizer)
