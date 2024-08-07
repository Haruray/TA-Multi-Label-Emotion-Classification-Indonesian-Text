from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer, T5Tokenizer
from tqdm import tqdm
import torch
import pandas as pd
from MLEC.dataset_processing.twitter_preprocessor import twitter_preprocessor


class DataClass(Dataset):
    """Class for processing dataset and preparing data for training.
    Args:
        filename (str): The path to the dataset file.
        max_length (int): The maximum length of the input sequences.
        tokenizer_name (str): The name of the tokenizer to be used.
        language (str, optional): The language of the dataset. Defaults to "Indonesia".
        use_prefix (bool, optional): Whether to use label names as prefix. Defaults to True.
    Methods:
        load_dataset():
            Loads and preprocesses the dataset.
        process_data():
            Processes the dataset and tokenizes the input sequences.
        __getitem__(index):
            Returns the data item at the specified index.
        __len__():
            Returns the length of the dataset."""

    def __init__(
        self,
        filename,
        max_length,
        tokenizer_name,
        language="Indonesia",
        use_prefix=True,
    ):
        self.language = language
        self.use_prefix = use_prefix
        self.filename = filename
        self.max_length = max_length
        self.data, self.labels, self.label_names = self.load_dataset()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # needed for span prediction
        self.tokenizer.add_tokens(self.label_names)

        (
            self.inputs,
            self.attention_masks,
            self.lengths,
            self.label_indices,
        ) = self.process_data()

    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        # if the file is a text file, then use sep="\t"
        if self.filename.endswith(".txt"):
            df = pd.read_csv(self.filename, sep="\t")
        else:
            # if the file is a csv file, then use sep=","
            df = pd.read_csv(self.filename, sep=",")
        x_train, y_train = df.Text.values, df.iloc[:, 2:].values
        # get label names
        label_names = df.columns[2:].tolist()
        return x_train, y_train, label_names

    def process_data(self):
        desc = "PreProcessing dataset {}...".format("")
        preprocessor = twitter_preprocessor(lang=self.language)

        segment_a = ""
        if self.use_prefix:
            segment_a = (" ".join(self.label_names) + "?").lower()

        (
            inputs,
            attention_masks,
            lengths,
            label_indices,
        ) = ([], [], [], [])

        for _, data_item in enumerate(tqdm(self.data, desc=desc)):
            data_item = " ".join(preprocessor(data_item))
            data_item = self.tokenizer.encode_plus(
                segment_a,
                data_item,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            input_id = data_item["input_ids"]
            attention_mask = data_item["attention_mask"]
            input_length = len([i for i in data_item["attention_mask"] if i == 1])

            inputs.append(input_id)
            lengths.append(input_length)
            attention_masks.append(attention_mask)

            # label indices
            label_idxs = []
            if self.use_prefix:
                label_idxs = [
                    self.tokenizer.convert_ids_to_tokens(input_id).index(
                        self.label_names[idx]
                    )
                    for idx, _ in enumerate(self.label_names)
                ]
            label_indices.append(label_idxs)

        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        return (
            inputs,
            attention_masks,
            data_length,
            label_indices,
        )

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        attention_masks = self.attention_masks[index]
        return (
            inputs,
            attention_masks,
            labels,
            length,
            label_idxs,
        )

    def __len__(self):
        return len(self.inputs)
