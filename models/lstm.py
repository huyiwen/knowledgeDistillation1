import torch
from torch import nn
from torch.nn import Embedding, GRU, LSTM
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel, PretrainedConfig, BertTokenizer
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from packaging import version

class LSTMTokenizer(BertTokenizer):
    pass


class LSTMConfig(PretrainedConfig):

    model_type = "lstm"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        intermediate_size=384,
        activated_hidden_size=384,
        hidden_dropout_prob=0.1,
        num_layers=2,
        bidirectional=True,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activated_hidden_size = activated_hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class LSTMPreTrainedModel(PreTrainedModel):

    config_class = LSTMConfig
    base_model_prefix = "lstm"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LSTMEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LSTMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        multiplier = 2 if config.bidirectional else 1
        self.dense = nn.Linear(config.intermediate_size * multiplier, config.activated_hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[0][:, -1, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LSTMModel(LSTMPreTrainedModel):

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = LSTMEmbeddings(config)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.intermediate_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        self.pooler = LSTMPooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None, **kwargs):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        lstm_output = self.lstm(embedding_output)
        pooler_output = self.pooler(lstm_output)
        return pooler_output


class LSTMForSequenceClassification(LSTMPreTrainedModel):

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__(config)
        self.config = config

        self.lstm = LSTMModel(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.activated_hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output = self.lstm(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.dropout(output)
        logits: torch.Tensor = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class _BiLSTM(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = Embedding(30522, 768)
        self.gru = GRU(
            input_size=768,
            hidden_size=384,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.head1 = nn.Linear(768, 384)
        self.head2 = nn.Linear(384, num_classes)
        raise DeprecationWarning("This model is deprecated. Use LSTMForSequenceClassification instead.")

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        x, _ = self.gru(x)
        x = self.head1(x[:, -1, :])
        x = self.head2(x)
        return x
