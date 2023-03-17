from torch import nn


class LSTMOutput(nn.Module):
    def __init__(self, out_len=1):
        super().__init__()
        self.out_len = out_len
        
    def forward(self,x):
        # nn.LSTM returns (output, (hn, cn)), so we just
        # want to grab the `output`
        # Output shape (batch, sequence_length, hidden)
        output, _ = x
        # Now just grab the last index on the sequence length
        # Reshape shape (batch, output_timesteps, hidden)
        return output[:, -self.out_len:, :]

    
def create_lstm_model(
    input_size, 
    hidden_size, 
    output_size, 
    output_sequence_length,
    num_layers, 
    dropout
):
    model = nn.Sequential(
        nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        ),
        LSTMOutput(output_sequence_length),
        nn.Linear(in_features=hidden_size, out_features=output_size),
        nn.LeakyReLU(),
    )
    return model