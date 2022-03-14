# %% [code] {"execution":{"iopub.status.busy":"2022-03-14T11:09:51.645405Z","iopub.execute_input":"2022-03-14T11:09:51.64609Z","iopub.status.idle":"2022-03-14T11:09:56.332893Z","shell.execute_reply.started":"2022-03-14T11:09:51.645999Z","shell.execute_reply":"2022-03-14T11:09:56.332111Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import json
import tensorflow as tf
import unicodedata
import re
import os
# visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score

from transformers import BertTokenizerFast, TFBertModel

# List the available devices available in the local process.

from tensorflow.python.client import device_lib
print( len(device_lib.list_local_devices()) )

df = pd.read_csv("../input/alldialectdataset/out.csv", encoding="utf-8", lineterminator='\n')
df.drop(columns=["id"], inplace=True)
df.columns = ['dialect', 'tweets']
df.dropna(inplace=True)

X = df.tweets
y = df.dialect

# `One Hot Encoder`: Encode categorical features as a one-hot numeric array.
ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(np.array(y).reshape(-1, 1)).toarray()

# Walkthrough all the texts in dataset to get the longset sequence of words in a text. Here we get 94 but we can add 6 to it for safety use.
#add = 6
MAX_LEN = max([len(x.split()) for x in X])# + add

# Set test size to 20 percentage of dataset. 
# * Raising the number of training dataset increasing the accuracy score.
test_size = 0.20

# # `BERT Model`
X_train, X_val, y_train, y_val = train_test_split(X, y_ohe, test_size=test_size, random_state=42)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def bert_tokenize(data, max_len=MAX_LEN):
    encoded = tokenizer(
        text = data.tolist(),
        add_special_tokens = True,
        max_length = MAX_LEN,
        truncation = True,
        padding = True, 
        return_tensors = 'tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True
    )
    return encoded['input_ids'], encoded['attention_mask']

train_input_ids, train_attention_masks = bert_tokenize(X_train, MAX_LEN)
val_input_ids, val_attention_masks = bert_tokenize(X_val, MAX_LEN)

# ### `callbacks`
# 
#     - Model Check point: Callback to save the Keras model or model weights at some frequency.
#     - Learning Rate Scheduler:  Updated learning rate value from schedule function 
#     - Early Stopping: Stop training when a monitored metric has stopped improving.

# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    """
        a function that takes an epoch index (integer, indexed from 0) 
        and current learning rate (float) as inputs 
        and returns a new learning rate as output (float).
    """
    if epoch < 10: return lr
    else: return lr * tf.math.exp(-0.1)
    
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "./bert_model_.h5", 
    #"./bert_model_{epoch:02d}-{val_categorical_accuracy:.4f}.h5', 
    monitor='val_categorical_accuracy', 
    verbose=1, save_best_only=True,
    mode='max'
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_categorical_accuracy",
    patience=5,
)

def create_model(bert_model, max_sequence=MAX_LEN):
    input_ids = tf.keras.layers.Input(shape=(max_sequence,), dtype=tf.int32, name='input_ids')
    #attention_mask = tf.keras.layers.Input((max_sequence,), dtype=tf.int32, name='attention_mask')
    #output = bert_model([input_ids, attention_mask])[0]
    output = bert_model([input_ids])[0]
    output = output[:, 0, :]
    output = tf.keras.layers.Dense(18, activation='softmax')(output)
    #model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
    model = tf.keras.models.Model(inputs=[input_ids], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)#3e-5
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss, metrics=accuracy)
    return model

model = create_model(bert_model)
history = model.fit(
    train_input_ids, 
    y_train,
    validation_data=(val_input_ids, y_val),
    batch_size=16,
    epochs=5,
    callbacks = [checkpoint, lr_scheduler, early_stopping]
)

test_loss, test_accuracy = model.evaluate(val_input_ids, y_val)

#print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_accuracy))

val_pred = model.predict(val_input_ids)

predicted = np.argmax(val_pred, axis=1)
y_val_encoded = np.argmax(y_val, axis=1)
predicted, y_val_encoded

accuracy_score(predicted, y_val_encoded) 
#precision_score(predicted, y_val_encoded, average='micro')

target_names = ohe.categories_[0]
target_names

cmap = sns.cubehelix_palette(as_cmap=True)
cm = confusion_matrix(y_val_encoded, predicted)
cm = pd.DataFrame(cm, target_names, target_names)
plt.figure(figsize = (20, 18))
sns.set(font_scale=1.4) # for label size
sns.heatmap(
    cm,
    fmt=".1f", 
    annot=True, 
    annot_kws={'size': 12},
    cmap = cmap,
    linewidths=3,
);
plt.show();

matrix = cm.diagonal() / cm.sum(axis=1)
pd.DataFrame({'Dialect': target_names[0], 'scores': matrix}).sort_values(by=['scores'], ascending=False)

rp = classification_report(
            predicted, 
            y_val_encoded,
            target_names = target_names, 
            digits=2,
            output_dict=True
        ),

report = pd.DataFrame(rp[0]).T.round(3)
report['support'] = report.support.apply(int)
report.round(3)

# # `Preprocessing and prediction for For Deployment`
# 
#     - load model
#     - bert tokenize inserted text
#     - pad sequence to maximun length (94)
#     - predict the padded sequence
#     - use numpy.argmax to extract the encode
#     - inverse one hot incoder transform to get predicted label.

def bert_tokenize_text(data, max_len=MAX_LEN) :
    input_ids = []
    attention_masks = []
    for tweet in data:
        encoded = tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=False
        )
        input_ids.append(encoded['input_ids'])
    return np.asarray(input_ids)#, np.asarray(attention_masks)

test = bert_tokenize_text(['كل اجة هتبقي كويسة'])
pred = model.predict(test)
pred = np.argmax(pred[0])

print(ohe.inverse_transform(pred)) # ['EG']
      
      
