import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformercl import TransformerClassifier
from transformerdecoder import TransformerLM
from utilities import Utilities  

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, attn_maps = model(inputs)  # Model returns outputs and attn_maps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """
    Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    losses = []

    decoderLMmodel.eval()
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        output, loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)
    decoderLMmodel.train()
    return perplexity


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)


    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

  
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    decoder_model = TransformerLM(tokenizer.vocab_size, n_embd, n_head, n_layer, n_hidden, n_output, block_size)



    encodermodel = TransformerClassifier(tokenizer.vocab_size, n_embd, n_head, n_layer, n_hidden, n_output, block_size)
    optimizer = torch.optim.AdamW(encodermodel.parameters(), lr=learning_rate)

    parser = argparse.ArgumentParser(description='Run specific part of the assignment')
    parser.add_argument('part', choices=['part1', 'part2'], help='Specify which part to run')
    args = parser.parse_args()
     # for the classification  task, you will train for a fixed number of epochs like this:
    if(args.part == 'part1'):
        num_params = count_parameters(encodermodel)
        optimizer = torch.optim.AdamW(encodermodel.parameters(), lr=learning_rate)

        print(f"Number of parameters in the encoder + classifier: {num_params}")
        for epoch in range(epochs_CLS):
            losses = torch.zeros((int) (len(train_CLS_loader)))
            for i, (xb, yb) in enumerate(train_CLS_loader):
                xb, yb = xb.to(device), yb.to(device)
                logits, loss = encodermodel(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses[i] = loss.item()
            train_loss = losses.mean()
            train_accuracy = compute_classifier_accuracy(encodermodel, train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(encodermodel, test_CLS_loader)
            print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        utilities = Utilities(tokenizer, encodermodel)
        sample_sentence = "Hello, this is a sample sentence for sanity check."
        with torch.no_grad():
            utilities.sanity_check(sample_sentence, block_size)
        torch.save(encodermodel.state_dict(), 'transformer_classifier.pth')


    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    elif(args.part == 'part2'):
        num_params = count_parameters(decoder_model)
        optimizer = torch.optim.AdamW(decoder_model.parameters(), lr=learning_rate)

        print(f"Number of parameters in the Decoder Model: {num_params}")

        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
        # LM training code here

            output, loss = decoder_model(xb, yb)
            optimizer.zero_grad()           
            loss.backward()

            optimizer.step()
            if i % eval_interval == 0:
                perplexity = compute_perplexity(decoder_model, train_LM_loader, eval_iters=eval_iters)
                print(f"Step {i}: Training Perplexity= {perplexity}")

        test_files = ["speechesdataset/test_LM_obama.txt", "speechesdataset/test_LM_wbush.txt", "speechesdataset/test_LM_hbush.txt"]
        test_names = ["Obama", "W. Bush", "H. Bush"]
        for test_file, test_name in zip(test_files, test_names):
            with open(test_file, 'r', encoding='utf-8') as f:
                test_text = f.read()
            test_LM_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
            test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=False)
            test_perplexity = compute_perplexity(decoder_model, test_LM_loader, eval_iters=eval_iters)
            print(f"Step {max_iters}: {test_name} Perplexity= {test_perplexity}")
        utilities = Utilities(tokenizer, decoder_model)
        sample_sentence = "Hello, this is a sample sentence for sanity check."
        with torch.no_grad():
            utilities.sanity_check(sample_sentence, block_size)
        torch.save(encodermodel.state_dict(), 'transformer_LM.pth')


    



if __name__ == "__main__":
    main()
