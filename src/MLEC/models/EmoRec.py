import torch.nn as nn
from MLEC.models.MLECModel import MLECModel
from transformers import AutoModel


class EmoRec(MLECModel):
    """
    Class for emotion recognition model.
    Args:
        output_dropout (float): Dropout rate for the output layer. Defaults to 0.1.
        alpha (float): Alpha parameter. Defaults to 0.2.
        beta (float): Beta parameter. Defaults to 0.1.
        embedding_vocab_size (int): Size of the embedding vocabulary. Defaults to 30522.
        label_size (int): Size of the label. Defaults to 8.
        device (str): Device to use for training. Defaults to "cuda:0".
        encoder_name (str): Name of the encoder model. Defaults to "indolem/indobert-base-uncased".
        name (str): Name of the model. Defaults to "emorec".
    Methods:
        forward(input_ids, input_attention_masks, targets=None, **kwargs):
            Forward pass of the model.
            Args:
                input_ids (Tensor): Input IDs.
                input_attention_masks (Tensor): Input attention masks.
                targets (Tensor, optional): Targets. Defaults to None.
                **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        output_dropout=0.1,
        alpha=0.2,
        beta=0.1,
        embedding_vocab_size=30522,
        label_size=8,
        device="cuda:0",
        encoder_name="indolem/indobert-base-uncased",
        name="emorec",
    ):
        super(EmoRec, self).__init__(alpha=alpha, beta=beta, device=device, name=name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.resize_token_embeddings(embedding_vocab_size)
        self.encoder.to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, label_size),
        ).to(device)

    def forward(self, input_ids, input_attention_masks, targets=None, **kwargs):
        lengths = kwargs.get("lengths", None)
        label_idxs = kwargs.get("label_idxs", None)
        input_attention_masks = input_attention_masks.to(self.device)
        input_ids, num_rows = input_ids.to(self.device), input_ids.size(0)

        if label_idxs is not None:
            label_idxs = label_idxs[0].long().to(self.device)

        if targets is not None:
            targets = targets.float().to(self.device)

        # Bert encoder
        output = self.encoder(
            input_ids=input_ids, attention_mask=input_attention_masks, return_dict=True
        )

        logits = self.classifier(output.pooler_output)

        y_pred = self.compute_pred(logits)

        return num_rows, y_pred, logits, targets
