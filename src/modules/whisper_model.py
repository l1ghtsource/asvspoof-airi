import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        # projection shortcut if dimensions don't match
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.linear1(x)
        out = self.gelu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.dropout(out)

        return self.gelu(out + identity)


class WhisperClassifier(nn.Module):
    def __init__(self, num_labels, encoder, dropout=0.1):
        super(WhisperClassifier, self).__init__()
        self.encoder = encoder
        self.dropout = dropout
        hidden_size = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            # first block
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2048),
            nn.GELU(),
            nn.Dropout(self.dropout),

            # residual block 1
            nn.LayerNorm(2048),
            ResidualBlock(2048, 2048),
            nn.Dropout(self.dropout),

            # residual block 2
            nn.LayerNorm(2048),
            ResidualBlock(2048, 1024),
            nn.Dropout(self.dropout),

            # final classification layers
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(512),
            nn.Linear(512, num_labels)
        )

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_features, decoder_input_ids):
        outputs = self.encoder(input_features, decoder_input_ids=decoder_input_ids)
        pooled_output = outputs['last_hidden_state'][:, 0, :]
        logits = self.head(pooled_output)
        return logits


class SimpleWhisperClassifierV1(nn.Module):
    def __init__(self, num_labels, encoder, dropout=0.1):
        super(SimpleWhisperClassifierV1, self).__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_features, decoder_input_ids):
        outputs = self.encoder(input_features, decoder_input_ids=decoder_input_ids)
        pooled_output = outputs['last_hidden_state'][:, 0, :]
        logits = self.head(pooled_output)
        return logits


class SimpleWhisperClassifierV2(nn.Module):
    def __init__(self, num_labels, encoder, dropout=0.1):
        super(SimpleWhisperClassifierV2, self).__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 4096),
            nn.LayerNorm(4096),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, num_labels),
        )

    def forward(self, input_features, decoder_input_ids):
        outputs = self.encoder(input_features, decoder_input_ids=decoder_input_ids)
        pooled_output = outputs['last_hidden_state'][:, 0, :]
        logits = self.classifier(pooled_output)
        return logits
