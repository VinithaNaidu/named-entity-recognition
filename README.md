# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
This project aims to implement an LSTM-based model for named entity recognition (NER) in text, targeting the identification of entities like persons, organizations, and locations. By leveraging deep learning techniques, we seek to develop a robust system capable of accurately labeling named entities in unstructured text data

## Dataset
![Screenshot 2024-10-04 111058](https://github.com/user-attachments/assets/955ab42a-0f55-4010-bf32-4a9758901755)


## DESIGN STEPS

### Step 1 : 
Import the necessary packages.

### Step 2 : 
Read the dataset, and fill the null values using forward fill.

### Step 3: 
Create a list of words, and tags. Also find the number of unique words and tags in the dataset.

### Step 4 : 
Create a dictionary for the words and their Index values. Do the same for the tags as well,Now we move to moulding the data for training and testing.

### Step 5 : 
We do this by padding the sequences,This is done to acheive the same length of input data.

### Step 6 : 
We build a build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.

### Step 7 : 
We compile the model and fit the train sets and validation sets,We plot the necessary graphs for analysis,A custom prediction is done to test the model manually.



## PROGRAM
```
Name: Vinitha D
Reg. No: 212222230175
```
```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data.head(50)
data = data.fillna(method="ffill")
data.head(50)
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())
print("Unique tags are:", tags)
num_words
getter = SentenceGetter(data)
sentences = getter.sentences
len(sentences)
sentences[0]

```

## OUTPUT :
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/bd0262f4-1e49-4919-b431-612788a33677)
![image](https://github.com/user-attachments/assets/79ef2b1d-5f49-4f76-a10e-503b6c258c11)

### Histogram Plot
![image](https://github.com/user-attachments/assets/b948b653-3dbd-4882-b54f-6cdd951c0a94)


### Sample Text Prediction
![image](https://github.com/user-attachments/assets/a98d19a5-9c59-4d17-b7e0-5b4e4badfb23)




## RESULT : 
Thus, an LSTM-based model (bi-directional) for recognizing the named entities in the text is developed Successfully.
