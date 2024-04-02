import torch
from torch import nn
import torch.nn.functional as F


def contrastive_loss(embeddings1, embeddings2, labels, margin=1.0):
    distance_matrix = F.pairwise_distance(embeddings1, embeddings2, p=2)
    # Positive samples
    positive_loss = labels.float() * distance_matrix ** 2
    # Negative samples
    negative_loss = (1.0 - labels.float()) * \
        torch.clamp((margin - distance_matrix), min=0) ** 2
    loss = 0.5 * (positive_loss + negative_loss).mean()
    return loss


class BertContrastive(nn.Module):
    def __init__(self, bert_model, hidden_size):
        super(BertContrastive, self).__init__()
        self.bert = bert_model
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None, return_embeddings=False):
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

        if return_embeddings:
            return u, v

        # If not returning embeddings, compute the contrastive loss
        loss = contrastive_loss(u, v, labels)
        return loss
