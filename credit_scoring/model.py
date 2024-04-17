import torch.nn as nn
import torch


class CreditsGRU(nn.Module):
    def __init__(self, features, embedding_projections, rnn_units=128, top_classifier_units=32, bidirectional=False):
        super(CreditsGRU, self).__init__()
        self.bidirectional = bidirectional
        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature])
                                                      for feature in features])

        # self.dropout2 = nn.Dropout2d(p=0.1)
        # self.dropout = nn.Dropout(p=0.1)

        self._gru = nn.GRU(input_size=sum([embedding_projections[x][1] for x in features]),
                            hidden_size=rnn_units, batch_first=True, bidirectional=bidirectional)
        self._hidden_size = rnn_units
        self._top_classifier = nn.Linear(
            in_features=rnn_units * 2 if self.bidirectional else rnn_units,
            out_features=top_classifier_units * 2 if self.bidirectional else top_classifier_units,
        )
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(
            in_features=top_classifier_units * 2 if self.bidirectional else top_classifier_units,
            out_features=1
        )
    
    def forward(self, features):
        batch_size = features[0].shape[0]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)

        # concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        # concated_embeddings = self.dropout2(concated_embeddings)
        # concated_embeddings = concated_embeddings.squeeze(3).permute(0, 2, 1)

        # print(f"input {concated_embeddings.shape}")
        pooling_emb = nn.AdaptiveMaxPool2d((1, 128))(concated_embeddings.permute(0, 2, 1)).squeeze()
        _, last_hidden = self._gru(concated_embeddings)
        # print(f"last hidden {last_hidden.shape}\n")

        # last_hidden = torch.reshape(
        #     last_hidden.permute(1, 2, 0),
        #     shape=(batch_size, self._hidden_size * 2 if self.bidirectional else self._hidden_size))

        pooling_gru = nn.AdaptiveMaxPool2d((1, 128))(last_hidden.permute(1, 0, 2)).squeeze()
        # print(f"emb pooling: {pooling_emb.shape}, LSTM pooling: {pooling_gru.shape}")

        pooling = torch.cat([pooling_emb, pooling_gru], dim=-1)

        # dropout_gru = self.dropout(pooling)
        classification_hidden = self._top_classifier(pooling)
        activation = self._intermediate_activation(classification_hidden)
        raw_output = self._head(activation)
        return raw_output
    
    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class CreditsLSTM(nn.Module):
    def __init__(self, features, embedding_projections, rnn_units=128, top_classifier_units=32, bidirectional=False):
        super(CreditsLSTM, self).__init__()
        self.bidirectional = bidirectional
        self._credits_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature])
                                                      for feature in features])

        self._lstm = nn.LSTM(input_size=sum([embedding_projections[x][1] for x in features]),
                           hidden_size=rnn_units, batch_first=True, bidirectional=bidirectional)
        self._hidden_size = rnn_units
        self._top_classifier = nn.Linear(
            in_features=rnn_units*2 if self.bidirectional else rnn_units,
            out_features=top_classifier_units*2 if self.bidirectional else top_classifier_units,
        )
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(
            in_features=top_classifier_units*2 if self.bidirectional else top_classifier_units,
            out_features=1
        )

    def forward(self, features):
        batch_size = features[0].shape[0]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)

        print(f"input {concated_embeddings.shape}")

        pooling_emb = nn.AdaptiveMaxPool2d((1, 128))(concated_embeddings.permute(0, 2, 1)).squeeze()

        _, (last_hidden, _) = self._lstm(concated_embeddings)

        print(f"last hidden {last_hidden.shape}\n")


        # last_hidden = torch.reshape(
        #     last_hidden.permute(1, 2, 0),
        #     shape=(batch_size, self._hidden_size*2 if self.bidirectional else self._hidden_size))

        # print(f"last hidden reshaped {last_hidden.shape}\n")

        pooling_lstm = nn.AdaptiveMaxPool2d((1, 128))(last_hidden.permute(1, 0, 2)).squeeze()

        print(f"emb pooling: {pooling_emb.shape}, LSTM pooling: {pooling_lstm.shape}")

        pooling = torch.cat([pooling_emb, pooling_lstm], dim=-1)

        classification_hidden = self._top_classifier(pooling)

        activation = self._intermediate_activation(classification_hidden)
        raw_output = self._head(activation)
        return raw_output

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality + add_missing, embedding_dim=embed_size, padding_idx=padding_idx)