from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F


def cosine_sim(a, b):
    dot_product = torch.mm(a, b.t())
    norm_a = a.norm(dim=1).unsqueeze(1)
    norm_b = b.norm(dim=1).unsqueeze(0)
    # Adding a small value to the denominator to prevent division by zero
    return dot_product / (norm_a * norm_b + 1e-7)


def eval_loop(model, tokenizer, eval_dataloader, device, model_type="bert"):
    model.eval()
    all_scores = []
    all_cosine_similarities = []

    sep_token_id = tokenizer.sep_token_id

    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scores = batch["score"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        with torch.no_grad():

            # Use the flag to determine output type
            if model_type == "classifier":
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                return_embeddings=True if model_type == "classifier" else False)
                embedding1, embedding2 = outputs
            elif model_type == "bert":
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                embeddings = outputs[0]
                sep_indices = (input_ids == sep_token_id).nonzero(
                    as_tuple=True)[1]
                embedding1 = embeddings[:, :sep_indices[0]].mean(dim=1)
                embedding2 = embeddings[:, sep_indices[0] +
                                        1:sep_indices[1]].mean(dim=1)
            elif model_type == "contrastive":
                embedding1, embedding2 = model(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_embeddings=True)
            elif model_type == "all_mpnet_base_v2":
                embeddings = model(
                    input_ids, attention_mask, token_type_ids)
                # ...

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            cosine_sim_values = cosine_sim(embedding1, embedding2).diagonal()
            all_cosine_similarities.extend(cosine_sim_values.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    eval_pearson_cosine = pearsonr(all_scores, all_cosine_similarities)[0]
    eval_spearman_cosine = spearmanr(all_scores, all_cosine_similarities)[0]

    return [eval_pearson_cosine, eval_spearman_cosine]

############ Evaluation for all_mpnet_base_v2 ############

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def eval_loop_v2(model, tokenizer, eval_dataloader, device, model_type="bert"):
    model.eval()
    all_scores = []
    all_cosine_similarities = []

    sep_token_id = tokenizer.sep_token_id

    for batch in eval_dataloader:
        if model_type == "all_mpnet_base_v2":
            input_ids_1 = batch["input_ids_1"].to(device)
            attention_mask_1 = batch["attention_mask_1"].to(device)
            input_ids_2 = batch["input_ids_2"].to(device)
            attention_mask_2 = batch["attention_mask_2"].to(device)
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
        scores = batch["score"].to(device)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.no_grad():

            # Use the flag to determine output type
            if model_type == "classifier":
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                return_embeddings=True if model_type == "classifier" else False)
                embedding1, embedding2 = outputs
            elif model_type == "bert":
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                embeddings = outputs[0]
                sep_indices = (input_ids == sep_token_id).nonzero(
                    as_tuple=True)[1]
                embedding1 = embeddings[:, :sep_indices[0]].mean(dim=1)
                embedding2 = embeddings[:, sep_indices[0] +
                                        1:sep_indices[1]].mean(dim=1)
            elif model_type == "contrastive":
                embedding1, embedding2 = model(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_embeddings=True)

            elif model_type == "all_mpnet_base_v2":
                embeddings_1 = model(input_ids=batch['input_ids_1'].to(
                    device), attention_mask=batch['attention_mask_1'].to(device))
                embeddings_2 = model(input_ids=batch['input_ids_2'].to(
                    device), attention_mask=batch['attention_mask_2'].to(device))

                sentence_embeddings_1 = mean_pooling(
                    embeddings_1, batch['attention_mask_1'].to(device))
                sentence_embeddings_2 = mean_pooling(
                    embeddings_2, batch['attention_mask_2'].to(device))

                # Normalize embeddings
                embedding1 = F.normalize(sentence_embeddings_1, p=2, dim=1)
                embedding2 = F.normalize(sentence_embeddings_2, p=2, dim=1)

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            cosine_sim_values = cosine_sim(embedding1, embedding2).diagonal()
            all_cosine_similarities.extend(cosine_sim_values.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    eval_pearson_cosine = pearsonr(all_scores, all_cosine_similarities)[0]
    eval_spearman_cosine = spearmanr(all_scores, all_cosine_similarities)[0]

    return [eval_pearson_cosine, eval_spearman_cosine]
