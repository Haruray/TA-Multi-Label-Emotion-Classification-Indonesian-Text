import torch.nn as nn
from transformers import AutoModel
from MLEC.models.MLECModel import MLECModel


class SpanEmo(MLECModel):
    """Class for training a SpanEmo model.
    Args:
        output_dropout (float): Dropout rate for the output layer. Defaults to 0.1.
        alpha (float): Alpha parameter for the model. Defaults to 0.2.
        beta (float): Beta parameter for the model. Defaults to 0.1.
        embedding_vocab_size (int): Size of the embedding vocabulary. Defaults to 30522.
        device (str): Device to run calculations on. Defaults to "cuda:0".
        encoder_name (str): Name of the encoder model. Defaults to "indolem/indobert-base-uncased".
        name (str): Name of the model. Defaults to "spanemo".
    Methods:
        forward(input_ids, input_attention_masks, targets=None, **kwargs):
            Forward pass of the model.
        compute_pred(logits):
            Computes the predictions based on the logits."""

    def __init__(
        self,
        output_dropout=0.1,
        alpha=0.2,
        beta=0.1,
        embedding_vocab_size=30522,
        device="cuda:0",
        encoder_name="indolem/indobert-base-uncased",
        name="spanemo",
    ):
        super(SpanEmo, self).__init__(
            alpha=alpha,
            beta=beta,
            device=device,
            name=name,
        )
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.resize_token_embeddings(embedding_vocab_size)
        self.encoder.to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.encoder.config.hidden_size, 1),
        ).to(device)

    def forward(self, input_ids, input_attention_masks, targets=None, **kwargs):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        lengths = kwargs.get("lengths", None)
        label_idxs = kwargs.get("label_idxs", None)
        input_attention_masks = input_attention_masks.to(self.device)
        input_ids, num_rows = input_ids.to(self.device), input_ids.size(0)
        if label_idxs is not None:
            label_idxs = label_idxs[0].long().to(self.device)

        if targets is not None:
            targets = targets.float().to(self.device)

        output = self.encoder(
            input_ids=input_ids, attention_mask=input_attention_masks, return_dict=True
        )
        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        logits = (
            self.classifier(output.last_hidden_state)
            .squeeze(-1)
            .index_select(dim=1, index=label_idxs)
        )

        y_pred = self.compute_pred(logits)
        return num_rows, y_pred, logits, targets
