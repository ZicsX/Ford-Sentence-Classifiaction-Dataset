# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from transformers import BertTokenizer
from transformers import TFBertModel

import warnings
warnings.filterwarnings('ignore')

# read test data
test=pd.read_csv('test_preprocessed.csv')

# Sentence_id
Sentence_id = test['Sentence_id']
test.drop('Sentence_id',axis=1,inplace=True)

# load model weights
model = tf.keras.models.load_model('ford-sentence-classifiaction')

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def generate_training_data(train, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(train['New_Sentence'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=256, 
            truncation=True, 
            padding='max_length', 
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks

# prepare test data
X_input_ids = np.zeros((len(test), 256))
X_attn_masks = np.zeros((len(test), 256))

X_input_ids, X_attn_masks = generate_training_data(test, X_input_ids, X_attn_masks, tokenizer)

# create a data pipeline for test data
test_dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks))

def datasetMapFunction(input_ids, attn_masks):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }

# create dataset with maping input ids and attention masks
test_dataset = test_dataset.map(datasetMapFunction)

# predict on test data
pred = model.predict(test_dataset)

# one hot to categorical
classes = ['Responsibility', 'Requirement', 'Skill', 'SoftSkill', 'Education',
           'Experience']

pred = np.argmax(pred, axis=1)

# create submission file
submission = pd.read_csv('sample_submission.csv')
submission['Sentence_id'] = Sentence_id
submission['Type'] = pred
submission['Type'] = submission['Type'].apply(lambda x: classes[x])
submission.to_csv('submission.csv', index=False)
