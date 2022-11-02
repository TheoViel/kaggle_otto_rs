import numpy as np
from transformers import AutoTokenizer
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast


def get_tokenizer(name, precompute=False, df=None, folder=None, add_special_tokens=False):
    if "deberta-v" in name:
        tokenizer_module = DebertaV2TokenizerFast
        if folder is None:
            if "v2" in name:
                folder = "../output/deberta-v2-xlarge/tokenizers"
            else:
                folder = "../output/deberta-v3-large/tokenizers"
    else:
        tokenizer_module = AutoTokenizer

    if folder is None:
        tokenizer = tokenizer_module.from_pretrained(name)
    else:
        tokenizer = tokenizer_module.from_pretrained(folder)

    tokenizer.name = name
    tokenizer.special_tokens = {
        "sep": tokenizer.sep_token_id,
        "cls": tokenizer.cls_token_id,
        "pad": tokenizer.pad_token_id,
    }

    if precompute:
        tokenizer.precomputed = precompute_tokens(df, tokenizer)
    else:
        tokenizer.precomputed = None

    if add_special_tokens:
        tokenizer.add_tokens(['\n', '\t', '\r'])  # commented

    return tokenizer


def precompute_tokens(df, tokenizer):
    feature_texts = df["feature_text"].unique()

    ids = {}
    offsets = {}

    for feature_text in feature_texts:
        encoding = tokenizer(
            feature_text,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        ids[feature_text] = encoding["input_ids"]
        offsets[feature_text] = encoding["offset_mapping"]

    texts = df["clean_text"].unique()

    for text in texts:
        encoding = tokenizer(
            text,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        ids[text] = encoding["input_ids"]
        offsets[text] = encoding["offset_mapping"]

    return {"ids": ids, "offsets": offsets}


def encodings_from_precomputed(feature_text, text, precomputed, tokenizer, max_len=300):
    tokens = tokenizer.special_tokens

    # Input ids
    if "roberta" in tokenizer.name:
        qa_sep = [tokens["sep"], tokens["sep"]]
    else:
        qa_sep = [tokens["sep"]]

    input_ids = [tokens["cls"]] + precomputed["ids"][feature_text] + qa_sep
    n_question_tokens = len(input_ids)

    input_ids += precomputed["ids"][text]
    input_ids = input_ids[: max_len - 1] + [tokens["sep"]]

    # Token type ids
    if "roberta" not in tokenizer.name:
        token_type_ids = np.ones(len(input_ids))
        token_type_ids[:n_question_tokens] = 0
        token_type_ids = token_type_ids.tolist()
    else:
        token_type_ids = [0] * len(input_ids)

    # Offsets
    offsets = [(0, 0)] * n_question_tokens + precomputed["offsets"][text]
    offsets = offsets[: max_len - 1] + [(0, 0)]

    # Padding
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([tokens["pad"]] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        offsets = offsets + ([(0, 0)] * padding_length)

    encoding = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "offset_mapping": offsets,
    }

    return encoding
