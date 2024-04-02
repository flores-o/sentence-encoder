import torch
import torch.nn as nn


class BertClassifier(nn.Module):
    def __init__(self, bert, hidden_size, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_labels),
            nn.LogSoftmax(dim=1)  # Added this line for softmax activation
        )

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        # print("Shape of input_ids in BertClassifier:", input_ids.shape)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]

        sep_positions = (input_ids == 102).nonzero(as_tuple=True)[1][::2]

        # List to collect processed u and v embeddings for the batch
        u_list = []
        v_list = []

        # Iterate over the batch
        for idx, sep_position in enumerate(sep_positions):
            u_embedding = last_hidden_state[idx, :sep_position, :].mean(
                dim=0, keepdim=True)  # Average pooling
            v_embedding = last_hidden_state[idx, sep_position:, :].mean(
                dim=0, keepdim=True)  # Average pooling

            u_list.append(u_embedding)
            v_list.append(v_embedding)

        # Stack the embeddings
        u = torch.cat(u_list, dim=0)
        v = torch.cat(v_list, dim=0)

        concat = torch.cat((u, v), 1)
        abs_diff = torch.abs(u - v)

        features = torch.cat((concat, abs_diff), 1)
        logits = self.classifier(features)

        if return_embeddings:
            return u, v

        return logits
