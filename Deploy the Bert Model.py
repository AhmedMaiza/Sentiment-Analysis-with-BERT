from pandas._libs.algos import take_2d_axis0_object_object
!pip install pandas
from google.colab import drive
import pandas as pd
import re
import string

drive.mount('/content/drive')
try:
    dfM = pd.read_csv('/content/drive/MyDrive/PFEFolder/codeacademygo.csv', sep=',', encoding='utf-8')
except UnicodeDecodeError:
    # If there is a decoding error, try reading the file with a different encoding
    try:
        dfM = pd.read_csv('/content/drive/MyDrive/PFEFolder/codeacademygo.csv', sep=',', encoding='iso-8859-1')
    except UnicodeDecodeError:
        print("Unable to read the file with any of the supported encodings.")


dfM['content'] = dfM['content'].apply(remove_emoji)
dfM['content'] = dfM['content'].apply(remove_punctuations)
dfM['content'] = dfM['content'].apply(convert_to_lowercase)
dfM['content'] = dfM['content'].apply(remove_special_chars)
dfM['content'] = dfM['content'].apply(remove_links)


# Load the saved model
import pandas as pd
import torch
from transformers import BertForSequenceClassification

batch_size = 1000
max_length = 128

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_path = '/content/drive/MyDrive/best_model_state_dict.bin'
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.load_state_dict(torch.load(model_path))

model.eval()
model.to(device)

predicted_labels = []
for i in range(0, len(dfM), batch_size):
    # Get the current batch
    chunk = dfM[i:i+batch_size]
    reviews = chunk['content'].tolist()
    review_tokens = [tokenizer.encode(review, max_length=max_length, truncation=True, padding='max_length') for review in reviews]
    review_inputs = torch.tensor(review_tokens)

    # Make predictions
    review_inputs = review_inputs.to(device)
    with torch.no_grad():
        outputs = model(review_inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Convert predictions to sentiment labels
    label_map = {0: 'negative', 1: 'positive'}
    predicted_labels_batch = [label_map[prediction.item()] for prediction in predictions]
    predicted_labels += predicted_labels_batch

# Add predicted sentiment to dataframe
dfM['sentiment'] = predicted_labels


dfM.to_csv("codeacademygo.csv", encoding='utf-8', index = False)
