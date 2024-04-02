import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_loop_bert_classifier(model, optimizer, train_dataloader, num_epochs, device):
    model.to(device)
    model.train()
    criterion = nn.NLLLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for input_ids, attention_mask, token_type_ids, labels in tqdm(train_dataloader):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(
                device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}")


def train_loop_bert_contrastive(model, optimizer, train_dataloader, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for input_ids, attention_mask, _, labels in tqdm(train_dataloader):
            optimizer.zero_grad()

            # Now, the model directly returns the loss
            loss = model(input_ids, attention_mask, labels.to(device))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}")
