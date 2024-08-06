import torch.nn as nn
from MLEC.models.MLECModel import MLECModel
from transformers import AutoModel


class EmoRec(MLECModel):

    def __init__(
        self,
        output_dropout=0.1,
        lang="English",
        alpha=0.2,
        beta=0.1,
        embedding_vocab_size=30522,
        label_size=8,
        device="cuda:0",
        encoder_name="indolem/indobert-base-uncased",
        name="emorec",
    ):
        """casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(EmoRec, self).__init__(alpha=alpha, beta=beta, device=device, name=name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.resize_token_embeddings(embedding_vocab_size)
        self.encoder.to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, label_size),
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

        # Bert encoder
        output = self.encoder(
            input_ids=input_ids, attention_mask=input_attention_masks, return_dict=True
        )

        logits = self.classifier(output.pooler_output)

        y_pred = self.compute_pred(logits)

        return num_rows, y_pred, logits, targets
