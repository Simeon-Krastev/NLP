# Question Answering via Reading Comprehension

##  Implement a Question-Answering Model

Using the pre-trained BERT model “bert-large-uncased-whole-word-masking-finetuned-squad” to answer these questions, the output of running on the Dev Set v2.0 returns the following metrics after the evaluation with the given code for SQuAD2.0:

```json
{
  "exact": 32.207529689210816,
  "f1": 36.421519124409265,
  "total": 11873,
  "HasAns_exact": 63.98448043184885,
  "HasAns_f1": 72.4245439548096,
  "HasAns_total": 5928,
  "NoAns_exact": 0.5214465937762826,
  "NoAns_f1": 0.5214465937762826,
  "NoAns_total": 5945
}
```
The implementation for this pre-trained model can be found in BERT_SQuAD_Model.py. A sliding window approach was used to deal with inputs tokenized into more than 512 tokens, as BERT cannot handle longer inputs than this. A custom implementation of a (BERT-based) question-answering model with the capacity to detect unanswerable questions can be found in Task 3.

## Unanswerable Question Detection
### Hypothesis
Aside from finding the best possible answer by calculating the span of text in which it can be found, it should be possible to include a probability score that the question is unanswerable and set a threshold for the prediction to output “unanswerable”. We can do this by training the model on unanswerable questions in addition to answerable ones and using a combined loss function that incorporates both the span prediction loss and a binary classification loss for unanswerability.

### Motivation
To make reading comprehension question & answer tasks more robust by identifying the answerability of questions related to a given text.

### Model Design
This architecture is based on a BERT (bert-base-uncased) transformer model as the backbone. The idea behind the model is to not only output answer span predictions, but also to classify if the question posed is answerable based on the given context.

Data is loaded and pre-processed from the SQuAD 2.0 dataset, which includes both answerable and unanswerable questions. Data samples are randomized and split into training, validation, and test sets, and the resulting test set is saved to use as ground truth input to the SQuAD evaluation script. BERT's base model calculates start and end logits for possible answers, and a separate linear layer predicts whether the question is answerable or not. This is done by outputting high probabilities through a sigmoid activation in the classification branch. Losses from both tasks (span prediction and binary classification) are combined in a custom function to guide the training of the model, which is fine-tuned on a validation set.

### Key Parameter Settings
- Optimizer: Adam with a learning rate of 5e-5
- Loss Functions: CrossEntropyLoss (span), BCEWithLogitsLoss (classification)
- Batch Size: 2 (due to memory constraints)
- Epochs: 8 (to avoid the model taking forever to train)

## Design Implementation
An implementation of this model can be found in Custom_SQuAD_Model.py. Due to hardware and time constraints, this model was trained on a small subset (15 paragraphs) of the Dev Set 2.0 data. After training, predictions were generated on the test split (484 questions):

```json
{
  "exact": 50.413223140495866,
  "f1": 54.68739981685866,
  "total": 484,
  "HasAns_exact": 12.033195020746888,
  "HasAns_f1": 20.617018719334375,
  "HasAns_total": 241,
  "NoAns_exact": 88.47736625514403,
  "NoAns_f1": 88.47736625514403,
  "NoAns_total": 243
}
```

This implementation improved the F1-Score of the predictions from 36.42% to 54.69%. Training with more data and more epochs could potentially improve this further.
