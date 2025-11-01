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

### STEP 4:
Define function for forward pass applying embedding, dropout, lstm and linear layer.

### STEP 5:
Return the output.

## PROGRAM
### Name: Cynthia Mehul J
### Register Number: 212223240020
```python
class BiLSTMTagger(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    super(BiLSTMTagger, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.dropout=nn.Dropout(0.1)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(hidden_dim * 2, output_dim)

  def forward(self, x):
    x=self.embedding(x)
    x=self.dropout(x)
    x,_=self.lstm(x)
    return self.fc(x)

model = BiLSTMTagger(len(word2idx) + 1, 100, 128, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=5):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

def evaluate_model(model, test_loader, X_test, y_test):
    model.eval()
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1)
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j] != tag2idx["O"]:
                        true_tags.append(idx2tag[labels[i][j].item()])
                        pred_tags.append(idx2tag[preds[i][j].item()])

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="487" height="117" alt="image" src="https://github.com/user-attachments/assets/7e9c8f39-859e-476e-949c-a357b47e7f10" />

<img width="729" height="621" alt="image" src="https://github.com/user-attachments/assets/ebd7e62f-b6d1-4348-ab51-bee55836ce90" />

### Sample Text Prediction

<img width="430" height="623" alt="image" src="https://github.com/user-attachments/assets/3a14cbbc-0e66-4d68-82c9-9bc45ac1ec51" />

## RESULT
Therefore, LSTM-based model for recognizing named entities in text is developed and executed successfully.
