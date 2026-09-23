from transformers import AutoModel
import torch.nn.functional as F
import torch
from torch import nn

from config import TOKEN
from utils import mean_pooling


class TraceClassifier(torch.nn.Module):
    def __init__(self, base_model: str, num_labels: int, clf: str = 'fc'):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model, token=TOKEN)
        self.clf_structure = clf

        # Optional LSTM classification head.
        if self.clf_structure.lower() == 'lstm':
            hidden_size = self.base_model.config.hidden_size # 768
            num_layers = 3
            self.lstm = nn.LSTM(input_size=hidden_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)

        self.classifier = torch.nn.Linear(self.base_model.config.hidden_size, num_labels)

        print(f'[TraceClassifier] base_model.config.hidden_size: {self.base_model.config.hidden_size}')
        print(f'[TraceClassifier] classifier structure: {self.clf_structure}')

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        if self.clf_structure.lower() == 'fc':
            embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.clf_structure.lower() == 'lstm':
            # Use the final LSTM step as the program embedding.
            lstm_out, (h_n, c_n) = self.lstm(outputs.last_hidden_state)  # lstm_out: (batch, seq_len, hidden_size)
            lstm_last_time_step = lstm_out[:, -1, :]  # (batch, hidden_size)
            embedding = lstm_last_time_step           # We only use the output of the last state of LSTM
        logits = self.classifier(embedding)
        return logits

class TraceClassifierForInference(TraceClassifier):
    def __init__(self, base_model: str, num_labels: int, clf: str = 'fc'):
        super().__init__(base_model, num_labels, clf)

    def forward(self, input_ids, attention_mask):
        # Get output class probabilities
        logits = super().forward(input_ids, attention_mask)
        pred_labels = torch.argmax(logits, dim=1)
        return logits, pred_labels

class TraceClassifierForAttribution(TraceClassifier):
    def __init__(self, base_model: str, num_labels: int, clf: str = 'fc'):
        super().__init__(base_model, num_labels, clf)

    def forward(self, input_ids, attention_mask):
        # Get output class probabilities
        logits = super().forward(input_ids.long(), attention_mask.long())
        return logits

class TraceClassifierForTraining(TraceClassifier):
    def __init__(self, base_model: str, num_labels: int, T: float = 0.07, clf: str = 'fc'):
        super().__init__(base_model, num_labels, clf)
        self.T = T
        self.loss_fn = torch.nn.CrossEntropyLoss()  # loss func used for one-hot classification

    def forward(self, input_ids, attention_mask, labels):
        # Get output class logits
        logits = super().forward(input_ids, attention_mask)

        # output_labels = torch.argmax(logits, dim=1)
        output_indices = torch.argmax(logits, dim=1)
        # transfer index to one-hot label
        num_classes = logits.size(1)  # get num of classes
        output_labels = nn.functional.one_hot(output_indices, num_classes=num_classes)

        # # debug
        # print(f'len of logits: {len(logits)}')
        # print(f'len of output_labels: {len(output_labels)}')
        # print(f'logits: {logits}')
        # print(f'output_labels: {output_labels}')
        # print(f'input_labels: {labels}')

        labels = labels.float()
        loss = self.loss_fn(logits, labels)
        return loss, logits, output_labels


if __name__ == '__main__':
    from dataset import ProgramDataset, create_collate_fn
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    model = TraceClassifierForTraining("microsoft/codebert-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    ds = ProgramDataset("../programs", "../labels.pt")
    dl = DataLoader(ds, batch_size=2, collate_fn=create_collate_fn(tokenizer))

    input_ids, attention_mask, labels = next(iter(dl))

    loss, logits = model(input_ids, attention_mask, labels)
    print(loss, logits)
