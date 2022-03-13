import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import json
import os

# Import dataset with text ids.
df = pd.read_csv("../input/dialect-datasetcsv/dialect_dataset.csv")
# create a new column to store feched text in
df['tweets'] = np.nan
# change 'id' column type to string.
df.id = df.id.astype(str)

# api-endpoint
URL = 'https://recruitment.aimtechnologies.co/ai-tasks'
start_sample, end_sample = 0, 0
num_samples = 1000
# Divide dataset length by maximum number of texts per request
#  to calculate the number of requests
quotient = len(df) // num_samples  # 458
remainder = len(df) % num_samples  # 197

# Run to get 1000 text and save them into our dataframe and save it 
def get_data(start_sample, num_samples):
    # calcualate end point of a request
    end_sample = start_sample + num_samples
    # get sample ids for POST
    sample = df.id[start_sample:end_sample].tolist()
    #print(sample[0])
    # sava sample as json
    with open('example.json', 'w') as txtfile:
        json.dump(sample, txtfile)
    # defining a params dict for the parameters to be sent to the API
    example = open('example.json', 'rb').read()
    # sending get request and saving the response as response object
    r = requests.post(url = URL, data = example)
    # extracting data in json format
    data = r.json()
    return data

# save requests texts in our dataset
def compine_data(df, post_data):
    d = pd.DataFrame.from_dict(post_data, orient='index')
    d.reset_index(inplace = True)
    d.columns = ['id', 'tweets']
    df = df.set_index('id').combine_first(d.set_index('id')).reset_index() # combine 
    return df

# make 458 'quotient' request -> quotient * num_samples = 458 * 1000
for i in range(quotient):
    start_sample = num_samples * i
    print(start_sample)
    #get 1000 sample and save it
    post_data = get_data(start_sample, num_samples)
    df = compine_data(df, post_data)

# get the rest of data in one request for 197 ids
if remainder != 0: 
    print('start remainder')
    start_sample = quotient * 1000
    #pass#print("get 1000 sample and save it")
    post_data = get_data(start_sample, remainder+1)
    # save to dataset
    df = compine_data(df, post_data)
    print("save last")
    
# SAVE OUR COMPLETE DF
file_name = 'out.csv'
df.to_csv(file_name, index=False, encoding="utf-8")
