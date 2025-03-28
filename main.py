import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems(file_name):
    """处理诗歌文本文件，并按长度分组"""
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line = line.strip()
                if ':' in line:
                    _, content = line.split(':')
                else:
                    content = line

                content = content.replace(' ', '')
                if any(char in content for char in ['_', '(', '（', '《', '[']) or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError:
                continue

    # 按长度分组
    len_to_poems = collections.defaultdict(list)
    for poem in poems:
        len_to_poems[len(poem)].append(poem)

    # 统计字符频率
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))

    return len_to_poems, word_int_map, words


def generate_batches(len_to_poems, word_to_int, batch_size):
    """生成相同长度的批次"""
    batches = []
    for length, poems in len_to_poems.items():
        poems_vec = [list(map(word_to_int.get, poem)) for poem in poems]
        n_batches = len(poems_vec) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = poems_vec[start:end]
            batch_y = [row[1:] + [row[-1]] for row in batch_x]
            batches.append((batch_x, batch_y, length))
    return batches


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, is_test=False):
        embed = self.embedding(x)
        lstm_out, _ = self.lstm(embed)
        output = self.fc(lstm_out)
        return nn.functional.log_softmax(output, dim=2)  # 明确指定dim


def run_training():
    """训练模型"""
    print("Loading data and initializing model...")
    len_to_poems, word_int_map, vocabularies = process_poems('./tangshi.txt')
    print(f"Loaded poems, vocabulary size: {len(word_int_map)}")

    BATCH_SIZE = 64
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    EPOCHS = 200

    torch.manual_seed(5)

    # 初始化模型
    vocab_size = len(word_int_map) + 1
    model = RNNModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    checkpoint_path = './poem_generator_full.pth'
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path))
            print("Loaded pre-trained model.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting fresh training.")

    print("Start training...")
    for epoch in tqdm(range(EPOCHS)):
        batches = generate_batches(len_to_poems, word_int_map, BATCH_SIZE)
        total_loss = 0

        for batch_x, batch_y, length in batches:
            x = torch.LongTensor(batch_x)
            y = torch.LongTensor(batch_y)

            optimizer.zero_grad()
            output = model(x)

            # 调整形状以适应损失函数
            output = output.transpose(1, 2)  # 从(batch, seq, vocab)变为(batch, vocab, seq)
            loss = criterion(output, y)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()

        torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch} completed. Avg loss: {total_loss / len(batches):.4f}")


def gen_poem(model, word_int_map, vocabularies, begin_word):
    """生成诗歌"""
    model.eval()
    if begin_word not in word_int_map:
        similar_words = [w for w in word_int_map if w.startswith(begin_word)]
        begin_word = similar_words[0] if similar_words else list(word_int_map.keys())[0]
        print(f"Using '{begin_word}' instead.")

    poem = begin_word
    word = begin_word
    max_length = 50

    with torch.no_grad():
        while word != end_token and len(poem) < max_length:
            input_seq = [word_int_map.get(w, 0) for w in poem]
            input_tensor = torch.LongTensor([input_seq])
            output = model(input_tensor, is_test=True)
            word = vocabularies[torch.argmax(output[0, -1])]
            poem += word

    return poem


if __name__ == "__main__":
    # 训练模型
    #run_training()

    # 加载模型和词汇表
    len_to_poems, word_int_map, vocabularies = process_poems('./tangshi.txt')
    vocab_size = len(word_int_map) + 1
    model = RNNModel(vocab_size, 100, 128)
    model.load_state_dict(torch.load('./poem_generator_full.pth'))

    # 生成诗歌
    starters = ["日", "红", "山", "夜", "湖", "海", "月", "君"]
    for starter in starters:
        poem = gen_poem(model, word_int_map, vocabularies, starter)
        print(f"\n生成以'{starter}'开头的诗:")
        print(poem.replace(start_token, '').replace(end_token, ''))
        print("-" * 40)