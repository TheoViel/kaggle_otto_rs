import torch
import transformers
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from utils.torch import load_pretrained_weights


class OttoTransformer(nn.Module):
    def __init__(
        self,
        model,
        nb_layers=1,
        nb_ft=None,
        use_lstm=False,
        use_conv=False,
        k=5,
        drop_p=0.1,
        multi_sample_dropout=False,
        num_classes=3,
        n_ids=2000000,
        config_file=None,
        pretrained=True,
        pretrained_weights=None,
        no_dropout=False,
    ):
        super().__init__()
        self.name = model
        self.n_ids = n_ids
        self.use_lstm = use_lstm
        self.nb_layers = nb_layers
        self.num_classes = num_classes
        self.multi_sample_dropout = multi_sample_dropout

        self.pad_idx = 1 if "roberta" in self.name else 0

        transformers.logging.set_verbosity_error()

        if config_file is None:
            config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        else:
            config = torch.load(config_file)

        if no_dropout:
            config.hidden_dropout_prob = 0
            config.attention_probs_dropout_prob = 0

        if pretrained:
            self.transformer = AutoModel.from_pretrained(model, config=config)
        else:
            self.transformer = AutoModel.from_config(config)

        self.transformer = update_embeds(self.transformer, n_ids, num_classes)

        self.nb_features = config.hidden_size
        if nb_ft is None:
            nb_ft = self.nb_features

        in_fts = self.nb_features * self.nb_layers

        if use_lstm:
            self.lstm = nn.LSTM(
                in_fts, self.nb_features, batch_first=True, bidirectional=True
            )
            in_fts = self.nb_features * 2

        if use_conv:
            self.cnn = nn.Sequential(
                nn.Conv1d(in_fts, nb_ft * 2, kernel_size=k, padding=k // 2),
                nn.Tanh(),
                nn.Dropout(drop_p),
                nn.Conv1d(nb_ft * 2, nb_ft, kernel_size=k, padding=k // 2),
                nn.Tanh(),
                nn.Dropout(drop_p),
            )
            self.logits = nn.Sequential(
                nn.Linear(nb_ft, num_classes * n_ids),
            )
        else:
            self.cnn = nn.Identity()
            self.logits = nn.Linear(in_fts, num_classes * n_ids)

        self.high_dropout = nn.Dropout(p=0.5)

        if pretrained_weights is not None:
            load_pretrained_weights(self, pretrained_weights)

    def forward(self, tokens, token_type_ids):
        """
        Usual torch forward function

        Arguments:
            tokens {torch tensor} -- Sentence tokens
            token_type_ids {torch tensor} -- Sentence tokens ids
        """

        if "distil" in self.name or "bart" in self.name:
            hidden_states = self.transformer(
                tokens,
                attention_mask=(tokens != self.pad_idx).long(),
            )[-1]
        else:
            hidden_states = self.transformer(
                tokens,
                attention_mask=(tokens != self.pad_idx).long(),
                token_type_ids=token_type_ids,
            )[-1]

        hidden_states = hidden_states[::-1]
        features = torch.cat(hidden_states[: self.nb_layers], -1)

        if self.use_lstm:
            features, _ = self.lstm(features)

        if self.multi_sample_dropout and self.training:
            logits = torch.mean(
                torch.stack(
                    [
                        self.logits(
                            self.cnn(
                                self.high_dropout(features).transpose(1, 2)
                            ).transpose(1, 2)[
                                :, 0
                            ]  # .mean(1)
                        )
                        for _ in range(5)
                    ],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.logits(
                self.cnn(features.transpose(1, 2)).transpose(1, 2)[:, 0]  # .mean(1)
            )

        return logits.view(-1, self.n_ids, self.num_classes)


def update_embeds(model, n_words, n_token_type, verbose=1):
    if verbose:
        print(f"Using {n_words} tokens for word_embeddings")
        print(f"Using {n_token_type} tokens for token_type_embeddings")

    model.config.type_vocab_size = n_token_type

    embedding_dim = model.embeddings.word_embeddings.embedding_dim
    padding_idx = model.embeddings.word_embeddings.padding_idx

    model.embeddings.word_embeddings = nn.Embedding(
        n_words, embedding_dim=embedding_dim, padding_idx=padding_idx
    )

    model.embeddings.token_type_embeddings = nn.Embedding(
        n_token_type, embedding_dim=embedding_dim
    )

    return model
