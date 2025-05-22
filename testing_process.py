# train_test_preprocess.py

from processing import pre_process

# Load your data
X, y, vocab = pre_process('test.csv', max_len=50)

# Print shapes or first sample to verify
print("Sample input (X[0]):", X[0])
print("Sample label (y[0]):", y[0])
print("Vocab size:", len(vocab))

from model import VanillaLSTM

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = 2  # binary classification
padding_idx = vocab['<PAD>']

model = VanillaLSTM(vocab_size=len(vocab), embedding_dim=100, hidden_dim=128, output_dim=2, padding_idx=vocab['<PAD>'])

# Check model output shape
sample_input = X[:4]  # batch of 4
print("Model output:", model(sample_input))
