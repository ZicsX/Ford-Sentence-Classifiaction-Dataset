# import libraries
import pandas as pd
import re

# read train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# preprocessiong

# remove null values
train = train.dropna()
test = test.dropna()

# remove HTML tags, Mail and URL

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_mail(text):
    clean = re.compile('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
    return re.sub(clean, '', text)

def remove_url(text):
    clean = re.compile('((www\.[^\s]+)|(https?://[^\s]+))')
    return re.sub(clean, '', text)

train['New_Sentence'] = train['New_Sentence'].apply(lambda x: remove_html_tags(x))
test['New_Sentence'] = test['New_Sentence'].apply(lambda x: remove_html_tags(x))

train['New_Sentence'] = train['New_Sentence'].apply(lambda x: remove_mail(x))
test['New_Sentence'] = test['New_Sentence'].apply(lambda x: remove_mail(x))
train['New_Sentence'] = train['New_Sentence'].apply(lambda x: remove_url(x))
test['New_Sentence'] = test['New_Sentence'].apply(lambda x: remove_url(x))

# remove words starts with '\\'
train['New_Sentence'] = train['New_Sentence'].apply(lambda x: re.sub(r'\\[a-zA-Z0-9]+', '', x))
test['New_Sentence'] = test['New_Sentence'].apply(lambda x: re.sub(r'\\[a-zA-Z0-9]+', '', x))

# remove words start with '/'
train['New_Sentence'] = train['New_Sentence'].apply(lambda x: re.sub(r'/[a-zA-Z0-9]+', '', x))
test['New_Sentence'] = test['New_Sentence'].apply(lambda x: re.sub(r'/[a-zA-Z0-9]+', '', x))

# replace '\' with ''
train['New_Sentence'] = train['New_Sentence'].apply(lambda x: re.sub(r'\\', '', x))
test['New_Sentence'] = test['New_Sentence'].apply(lambda x: re.sub(r'\\', '', x))

# remove b'
train['New_Sentence'] = train['New_Sentence'].apply(lambda x: re.sub(r"b'", '', x))
test['New_Sentence'] = test['New_Sentence'].apply(lambda x: re.sub(r"b'", '', x))

# remove multiple spaces
train['New_Sentence'] = train['New_Sentence'].apply(lambda x: re.sub(r'\s+', ' ', x))
test['New_Sentence'] = test['New_Sentence'].apply(lambda x: re.sub(r'\s+', ' ', x))

train['New_Sentence'] = train['New_Sentence'].apply(lambda x: x.strip())
test['New_Sentence'] = test['New_Sentence'].apply(lambda x: x.strip())

# save preprocessed data
train.to_csv('train_preprocessed.csv', index=False)
test.to_csv('test_preprocessed.csv', index=False)
