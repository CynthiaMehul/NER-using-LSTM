# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Identification of named entities is a major task in Natual Language Processing. To perform Named Entity Recognition, BiDirectional LSTM (Long Short Term Memory) can be used. The dataset used has various sentences along with its words and the corresponding tags. The task is to build a model that can classify these words into the right tags.

## DESIGN STEPS

### STEP 1:
Import required libraries in python to build your model and train it. Load your dataset and store required fields into different variables.

### STEP 2:
Allocate indexes for each unique word. Groud the words of each sentences and add the unique indexes for the words of each sentence. Split the dataset for training and testing. 

### STEP 3:
Build your BiDirectional LSTM Model. It consists of embedding layer to convert the indexes of each word into vectors. Use dropout method to prevent overfitting. Specify the lstm and linear layer. 

## PROGRAM
### Name:
### Register Number:
```python
class BiLSTMTagger(nn.Module):
    # Include your code here







    def forward(self, input_ids):
        # Include your code here
        


model = 
loss_fn = 
optimizer = 


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    # Include the training and evaluation functions






    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

Include your plot here

### Sample Text Prediction
Include your sample text prediction here.

## RESULT
