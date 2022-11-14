# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from transformers import BertTokenizer

import warnings
warnings.filterwarnings('ignore')

# read test data
test=pd.read_csv('test_preprocessed.csv')
test.fillna('Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime mollitia,molestiae quas vel sint commodi repudiandae consequuntur voluptatum laborum', inplace=True)

# Sentence_id
Sentence_id = test['Sentence_id']
test.drop('Sentence_id',axis=1,inplace=True)

# load model weights
model = tf.keras.models.load_model('ford-sentence-classifiaction')

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# one hot to categorical
classes = ['Responsibility', 'Requirement', 'Skill', 'SoftSkill', 'Education',
           'Experience']

def prepare_data(input_text):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256, 
        truncation=True, 
        padding='max_length', 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(input_text):
    processed_data = prepare_data(input_text)
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]

# make predictions
pred = []
for i in tqdm(range(len(test))):
    pred.append(make_prediction(test['New_Sentence'][i]))

# create submission file
submission = pd.DataFrame({'Sentence_id': Sentence_id, 'Type': pred})
submission.to_csv('submission.csv', index=False)
