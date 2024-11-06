from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.ln(x)
        return x
