## Arabic Dialect Classification



### Introduction

Arabic has several diverse dialects from across different regions of the Arab World. Although primarily spoken, written dialectal Arabic has been increasingly used on social media. Here we use collected tweets with 18 different Arabic dialects to train machine learning and deep learning models to classify these dialects.



### Data fetching 

To get the dataset, we’re using the “id” column in dialect_dataset.csv to retrieve the texts by calling [this API](https://recruitment.aimtechnologies.co/ai-tasks) by a POST request. By creating a JSON file as a request body containing a list of strings, the list’s size must not exceed 1000. So the solution used to break the large into several requests (number of IDs // 1000 = 458 requests) plus one more request for the remainder (197 IDs).




### Data exploration & preprocessing
Using bar plot to highlight some of the noisy words that spread in all dialects 

and create preprocessing pipeline to try several methods one at a time to clean our tweets like:-

    * Select Arabic characters only from text

    * Remove username “@handle” from text

    * Remove URL from text

    * Remove punctuation, emoji, and smileys from text

    * Remove \n, \t ,,, etc from text

    * Remove Discretization from text

    * Remove Arabic Stop Words from the text
    
The dataset is imbalanced as shown in the next pie plot. 
![](/images/pie.png)

So we can use oversampling or undersampling methods. 

The oversampling method helps with low text classes like ‘YE’.

Trying to combine a group of shared border countries dialects 

and run a classifier for first look investigation. 

We can try to reallocate dialects in different regions for a better accuracy score.

![](/images/cm.png)

Here we can Combine dialects with dialects for share border countries. So let’s take a small look here.

Starting with defining our new regions for dialects




### Machine Learning Models Training
After testing previous pre-processing methods on some machine learning models, 

Only removing Discretization and Escape Codes makes a slight improvements light in the score.

Next, convert the collection of text documents to a matrix of token counts

then transform the matrix to a normalized tf or tf-idf representation to be ready for the training stage. 

For target classes, Use Label Encoder to encode target labels with a value between 0 and the number of classes - 1 (0 to 17).

The next step is to divide the data set into training and validation data sets at 80% and 20%.

For large datasets consider using:-

  - linear support vector machine (SVM)

  - Logistic Regression 

  - Regularized linear models with stochastic gradient descent (SGD) learning “SGD Classifier”

As the classes are 18 we can use One Vs. Rest also referred to as the One-vs-All method 

for using binary classification algorithms for multi-class classification.

It involves splitting the multi-class dataset into multiple binary classification problems.

Scores at this point are:-

  - LinearSVC = 0.722

  - Logistic Regression = 0.703

  - SGD Classifier  = 0.657

Use Confusion Matrix and classification report to visualize precision and recall for each model.

Let’s focus in high misclassification ratio happens for each dialect:-

- `AE` :  SA, KW, BH
- `BH` :  AE, KW, OM, SA, QA
- `DZ` :  MA, TN, LY
- `EG` :  LY, PL, SD
- `IQ` :  KW, BH
- `JO` :  PL, LB, SY
- `KW` :  BH, AE, QA, SA
- `LB` :  PL, SY
- `LY` :  EG, KW, PL, TN
- `MA` :  DZ, LY
- `OM` :  AE, BH, KW, QA, SA
- `PL` :  JO, EG, LB, LY, SY
- `QA` :  SA, KW, AE, BH
- `SA` :  QA`, KW, BH, AE
- `SD` :  EG, LY, PL
- `SY` :  LB, PL
- `TN` :  LY, DZ
- `YE` :  SA

Most misclassifications happen for each dialect with dialects for share border countries. So we can make groups of share border countries like:-

- [‘QA’, ‘SA’, ‘AE’, ‘KW’, ‘OM’, ‘YE’,’BH’]

- [‘LY’, ‘SY’, ‘JO’, ‘PL’, ‘IQ’]

- [‘EG’, ‘LY’, ‘SD’]

- [‘TN’, ‘MA’, ‘DZ’]

By breaking down accuracy scores to individual classes. If a model has higher accuracy scores for some classes than another model and vice versa we can use Voting Classifier to get better overall accuracy.

Final step in this phase is to use oversampling method to upraise the number texts for dialects like ‘YE’

Scores after oversampling method are:-

  - LinearSVC = 0.713

  - Logistic Regression = 0.707

  - SGD Classifier  = 0.642


### Deep Learning Models Training

ur first step is to run string preprocessing and tokenize our dataset. 
This can be done using the 'BertTokenizer', 
Which is a text splitter that can tokenize sentences into subwords or word pieces for the BERT model
Given a vocabulary generated by the Word piece algorithm.

The texts are tokenized using Word Piece and a vocabulary size of 30,000. 
The inputs of the model are then of the form:

> `[CLS] Sentence A [SEP] Sentence B [SEP]`
![](https://miro.medium.com/max/1400/0*m_kXt3uqZH9e7H4w.png)

Now that we have all the inputs for our model,
The last step in our preprocessing is to package them into fixed 2-dimensional Tensors with padding

Padding to the maximum number of words through all texts (94).
We can add 6 more 

For target classes, Use one-hot Encoder to encode categorical features as a one-hot numeric array.

The next step is to divide the data set into training and validation data sets at 80% and 20%.


[bert-large-uncased](https://huggingface.co/bert-large-uncased)


Score:-
   - BRET: 0.731

### deployment
Use Flask and Plotly to visualize prediction probabilities for the input text.

![](/deploy_flask_plotly/Final Look/image 1.png)
![](/deploy_flask_plotly/Final Look/image 2.png)



