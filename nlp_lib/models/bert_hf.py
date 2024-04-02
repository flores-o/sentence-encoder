from transformers import BertConfig, AutoModel
import os
import shutil
import torch


# Constants
BERT_CONFIG = {
    "vocab_size": 30522,
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 512,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02
}

RENAMED_WEIGHTS_PATH = "renamed_weights.bin"


def adapt_weights_for_hf_bert(input_path, output_path):
    """ Rename the weights to be compatible with Huggingface transformers. """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist.")

    weights = torch.load(input_path)
    renamed_weights = rename_weights(weights)

    torch.save(renamed_weights, output_path)


def rename_weights(weights):
    """ Handles the logic for renaming weights. """

    # Dictionary to store the renamed weights
    renamed_weights = {}

    # Rename weights for embeddings
    renamed_weights["embeddings.word_embeddings.weight"] = weights["embeddings.token.weight"]
    renamed_weights["embeddings.position_embeddings.weight"] = weights["embeddings.position.weight"]
    renamed_weights["embeddings.token_type_embeddings.weight"] = weights["embeddings.token_type.weight"]
    renamed_weights["embeddings.LayerNorm.weight"] = weights["ln.gamma"]
    renamed_weights["embeddings.LayerNorm.bias"] = weights["ln.beta"]

    # Rename weights for attention layers and intermediate layers for both layers (0 and 1)
    for i in range(2):
        prefix = f"encoder.layer.{i}"
        renamed_weights[f"{prefix}.attention.self.query.weight"] = weights[f"layers.{i}.query.weight"]
        renamed_weights[f"{prefix}.attention.self.query.bias"] = weights[f"layers.{i}.query.bias"]
        renamed_weights[f"{prefix}.attention.self.key.weight"] = weights[f"layers.{i}.key.weight"]
        renamed_weights[f"{prefix}.attention.self.key.bias"] = weights[f"layers.{i}.key.bias"]
        renamed_weights[f"{prefix}.attention.self.value.weight"] = weights[f"layers.{i}.value.weight"]
        renamed_weights[f"{prefix}.attention.self.value.bias"] = weights[f"layers.{i}.value.bias"]
        renamed_weights[f"{prefix}.attention.output.dense.weight"] = weights[f"layers.{i}.attn_out.weight"]
        renamed_weights[f"{prefix}.attention.output.dense.bias"] = weights[f"layers.{i}.attn_out.bias"]
        renamed_weights[f"{prefix}.attention.output.LayerNorm.weight"] = weights[f"layers.{i}.ln1.gamma"]
        renamed_weights[f"{prefix}.attention.output.LayerNorm.bias"] = weights[f"layers.{i}.ln1.beta"]
        renamed_weights[f"{prefix}.intermediate.dense.weight"] = weights[f"layers.{i}.mlp.dense_expansion.weight"]
        renamed_weights[f"{prefix}.intermediate.dense.bias"] = weights[f"layers.{i}.mlp.dense_expansion.bias"]
        renamed_weights[f"{prefix}.output.dense.weight"] = weights[f"layers.{i}.mlp.dense_contraction.weight"]
        renamed_weights[f"{prefix}.output.dense.bias"] = weights[f"layers.{i}.mlp.dense_contraction.bias"]
        renamed_weights[f"{prefix}.output.LayerNorm.weight"] = weights[f"layers.{i}.ln2.gamma"]
        renamed_weights[f"{prefix}.output.LayerNorm.bias"] = weights[f"layers.{i}.ln2.beta"]

    # Rename weights for pooler
    renamed_weights["pooler.dense.weight"] = weights["pooler.dense.weight"]
    renamed_weights["pooler.dense.bias"] = weights["pooler.dense.bias"]

    return renamed_weights


def get_bert_config():
    """ Returns a BertConfig instance with custom configurations. """
    return BertConfig(**BERT_CONFIG)


def setup_config_and_copy_weights(config, weights_path, config_directory_path):
    """ Saves the config and copies the weights to a specified directory. """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} does not exist.")

    config.save_pretrained(config_directory_path)

    new_name = os.path.join(config_directory_path, "pytorch_model.bin")
    shutil.copy(weights_path, new_name)


def load_custom_configured_hf_bert(weights_path, device, config_directory_path="./custom_bert_directory"):
    """ Load a BERT model with a custom configuration. """
    config = get_bert_config()

    # Ensure weights are in the correct format
    adapt_weights_for_hf_bert(weights_path, RENAMED_WEIGHTS_PATH)

    # Save configuration and copy the weights
    setup_config_and_copy_weights(
        config, RENAMED_WEIGHTS_PATH, config_directory_path)

    # Load and return the model
    return AutoModel.from_pretrained(config_directory_path).to(device)
