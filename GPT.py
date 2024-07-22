import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the GPT model
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dropout=0.1):
        super(SimpleGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, embed_size))
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_size, num_heads, dim_feedforward=2048, dropout=dropout) 
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.fc_out(x)
        return logits

# Hyperparameters
vocab_size = 10000  # Adjust based on your dataset
embed_size = 256
num_heads = 8
num_layers = 6

model = SimpleGPT(vocab_size, embed_size, num_heads, num_layers)
class TextDataset(Dataset):
    def __init__(self, text, seq_length, vocab_size):
        self.text = text
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.char_to_idx = {ch: i for i, ch in enumerate(sorted(set(text)))}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.text[idx:idx+self.seq_length+1]
        input_text = torch.tensor([self.char_to_idx[ch] for ch in chunk[:-1]], dtype=torch.long)
        target_text = torch.tensor([self.char_to_idx[ch] for ch in chunk[1:]], dtype=torch.long)
        return input_text, target_text

# Sample text
text = "your dataset text here"
seq_length = 32

dataset = TextDataset(text, seq_length, vocab_size)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_text, target_text) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(input_text)
            loss = criterion(output.view(-1, vocab_size), target_text.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

# Train the model
train(model, dataloader, optimizer, criterion, epochs=10)
def generate_text(model, start_text, char_to_idx, idx_to_char, length):
    model.eval()
    input_text = torch.tensor([char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0)
    generated_text = start_text

    with torch.no_grad():
        for _ in range(length):
            output = model(input_text)
            next_char_idx = torch.argmax(output[:, -1, :], dim=-1).item()
            next_char = idx_to_char[next_char_idx]
            generated_text += next_char
            input_text = torch.cat([input_text, torch.tensor([[next_char_idx]], dtype=torch.long)], dim=1)

    return generated_text

# Generate text
start_text = "your start text"
generated_text = generate_text(model, start_text, dataset.char_to_idx, dataset.idx_to_char, length=100)
print(generated_text)
