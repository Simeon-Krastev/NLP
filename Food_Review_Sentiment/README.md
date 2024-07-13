# Semantic Analysis via Text Classification

In this exercise, text classification problems were re-investigated using content embedding techniques such as Word2Vec and BERT models. A review sentiment classification task was implemented using the Amazon Fine Food Reviews dataset, with a focus on a 10-20% sample. The steps included installing necessary packages, performing exploratory data analysis, creating labels based on review scores, and balancing the dataset. Various models were implemented: 
* TFIDF-based classifier
* Word2Vec-based classifier
* Pre-trained BERT classifier
* Fine-tuned BERT classifier
* BERT classifier with LoRA.

Finally, the results of these models were analyzed and compared.

The following results were compiled after downsampling the original Amazon Reviews dataset to balance the number of positive and negative labels as well as further sampling 15% of the data for easier computation. This amounts to 19,688 training and 4,923 testing examples. The code was executed using Google Colab; the link to the notebook can be found [here](https://drive.google.com/file/d/1u-hcDF9UzQVQbDrhStBg1McvNDXL8g6n/view?usp=sharing).

The code was executed in batches due to the timeout limits on Google Colab, as it was impossible to get the entire notebook to run without the runtime disconnecting at some point.

## Results Analysis
The implementation of each model alongside its individual results can be found in the accompanying .ipynb file.

### Results

| Method                       | Accuracy | Precision | Recall  | F1 Score |
|------------------------------|----------|-----------|---------|----------|
| TFIDF Logistic Regression    | 0.851920 | 0.860494  | 0.838367| 0.849287 |
| TFIDF SVM                    | 0.848263 | 0.856425  | 0.835102| 0.845629 |
| TFIDF GBM                    | 0.848263 | 0.856425  | 0.835102| 0.845629 |
| Word2Vec Logistic Regression | 0.780825 | 0.777643  | 0.783673| 0.780646 |
| Word2Vec Random Forest       | 0.784887 | 0.777202  | 0.795918| 0.786449 |
| Word2Vec SVM                 | 0.779809 | 0.779855  | 0.779774| 0.779782 |
| Pre-trained BERT             | 0.832419 | 0.873907  | 0.775102| 0.821544 |
| Fine-Tuned BERT              | 0.923014 | 0.936736  | 0.906531| 0.921386 |
| BERT with LoRA               | 0.911436 | 0.925973  | 0.893469| 0.909431 |

### Analysis

#### TFIDF-Based Models

These models were quite straightforward to implement. As in the first assignment, these models all use TFIDF vectors as features, capturing word importance based on their frequency in reviews relative to their frequency across the entire dataset. This mostly corresponds to identifying keywords that can indicate a certain target class (positive/negative reviews), although it fails to always take contextual information into account.

#### Word2Vec-Based Models

The three Word2Vec models with POS tagging that focus on adjectives and nouns show lower performance compared to the TFIDF-based models. Since Word2Vec mainly focuses on capturing semantic similarities between words rather than the specific presence or absence of terms, simple classifiers like logistic regression may not fully model the semantically-aware representations provided by Word2Vec.

#### BERT-Based Models

- **Pre-trained BERT**: Without fine-tuning, it performs well as a transformer model, even through the use of generic pre-training parameters. Its performance is comparable to TFIDF-methods, but offers a baseline with much room for fine-tuning based on the given dataset.
- **Fine-Tuned BERT**: Significantly outperforms other models, but it was computationally expensive, taking several hours to complete three epochs of training. This seems to be the default number of epochs, and experimenting with more might lead to overfitting but is still a meaningful exercise to try given more time.
- **BERT with LoRA**: Almost on par with the Fine-Tuned BERT model, but its usage of LoRA allows it to reduce the number of trainable parameters, producing results much faster than training all BERT parameters from scratch. In this case, 6 epochs trained in barely above one hour. This shows LoRA as a valid, resource-efficient alternative to a fully-trained BERT model.

### References

Fadheli, A. (2023, May). How to fine-tune BERT for text classification using Transformers in Python. [Blog post]. Retrieved from [https://thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python](https://thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python)

Karkar, N. (2023, October 14). Fine-tuning BERT for text classification with LoRA. [Blog post]. Retrieved from [https://medium.com/@karkar.nizar/fine-tuning-bert-for-text-classification-with-lora-f12af7fa95e4](https://medium.com/@karkar.nizar/fine-tuning-bert-for-text-classification-with-lora-f12af7fa95e4)

