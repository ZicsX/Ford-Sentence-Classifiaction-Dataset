# Ford Sentence Classification

In this challenge, the goal is to classify a sentence into one of the following categories:

- Responsibility
- Requirement
- Skill
- SoftSkill
- Education
- Experience

I have used BERT Model to classify the sentences. The model is trained on 80% of the data and tested on the remaining 20% of the data. The model is saved as 'ford-sentence-classifiaction' in the directory.
I have used BERT base model (cased) which is Pretrained model on English language using a masked language modeling (MLM) objective

In preprocessing, I have used the following steps to clean the data:

- Remove Null Values
- Remove all HTML tags, Mail and URL
- Remove all the single characters from the start
- Remove emoticons and emojis
- Remove all the special characters
- Remove all single characters from the start
- Substituting multiple spaces with single space
- Remove all the single characters from the start

I have used the following steps to tokenize the data:

- Tokenize the sentences
- Pad and truncate all the sentences to a maximum length of 256
- Create input ids and attention masks

I have used the following steps to train the model:

- Set the maximum length of the sentence to 256
- Set the batch size to 16
- Set the number of epochs to 5 
- Set the learning rate to 1e-5, decay=
- Set the epsilon to 1e-6
- Set the AdamW optimizer

### Installation

- Python
- Pandas
- Numpy
- Tensorflow
- Transformers
