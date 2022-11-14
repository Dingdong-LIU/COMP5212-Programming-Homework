import torch
import torch.nn.functional as F

class BidirectionalLSTM(torch.nn.Module):

    def __init__(self, embedding_dim, lstm_hidden_dim, fc_hidden_dim, vocab, output_size) -> None:
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        vocab_size = len(vocab)
        self.word_embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=vocab["<pad>"]
        )

        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim, hidden_size=self.lstm_hidden_dim, bidirectional=True, batch_first=True
        )
        # self.lstm2 = torch.nn.LSTM(input_size=)

        self.lstm2fc = torch.nn.Linear(2*self.lstm_hidden_dim, fc_hidden_dim)
        self.fc2label = torch.nn.Linear(fc_hidden_dim, output_size)


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, (h_state, c_state) = self.lstm(embeds.view(len(sentence), -1, self.embedding_dim))
        lstm_out_forward = lstm_out[:, -1, :self.lstm_hidden_dim]
        lstm_out_backward = lstm_out[:, -1, self.lstm_hidden_dim:]
        out_reduced = torch.concat((lstm_out_forward, lstm_out_backward), dim=1)
        # h_state = h_state.view(-1, 2*self.hidden_dim)
        # tag_space = self.hidden2label(lstm_out.view(len(sentence), -1))
        fc_out = torch.sigmoid(self.lstm2fc(out_reduced))
        score = self.fc2label(fc_out)
        return score

