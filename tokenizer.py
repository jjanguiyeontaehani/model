# tokenizer.py
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

def default_config():
    return {
        "vocab_size": 10000,
        "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        "corpus_path": "depression_dataset.txt",
        "model_path": "my_wordpiece_tokenizer.json",
        "single_template": "[CLS] $A [SEP]",
        "pair_template": "[CLS] $A [SEP] $B [SEP]",
        "max_length": 512,
    }

class myTokenizer:
    def __init__(self, config = default_config()):
        self.vocab_size = config["vocab_size"]
        self.special_tokens = config["special_tokens"]
        self.corpus_path = config["corpus_path"]
        self.model_path = config["model_path"]
        self.single_template = config["single_template"]
        self.pair_template = config["pair_template"]
        self.max_length = config["max_length"]

    def train_tokenizer(self):
        """
        Trains a WordPiece tokenizer on the provided corpus.
        :return: Trained tokenizer object.
        """
        tokenizer = Tokenizer(models.WordPiece(unk_token=self.special_tokens[1]))
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )

        pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
        )

        tokenizer.pre_tokenizer = pre_tokenizer
    
        tokenizer.decoder = decoders.WordPiece()

        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,
        )


        tokenizer.train([self.corpus_path], trainer=trainer)
        tokenizer.post_processor = processors.TemplateProcessing(
            single=self.single_template,
            pair=self.pair_template,
            special_tokens=[(self.special_tokens[2], 0), (self.special_tokens[3], 1)],
        )


        tokenizer.save(self.model_path)
        print(f"Tokenizer trained and saved to {self.model_path}")
        return tokenizer
    
    def encode(self, text):
        """
        Encodes the input text into token IDs.
        :param text: The input text to encode.
        :return: List of token IDs.
        """
        tokenizer = Tokenizer.from_file(self.model_path)
        if (self.max_length > 0):
            tokenizer.enable_truncation(max_length=self.max_length)
        encoded = tokenizer.encode(text)
        return encoded

    def decode(self, token_ids):
        """
        Decodes the list of token IDs back into text.
        :param token_ids: List of token IDs to decode.
        :return: The decoded text.
        """
        tokenizer = Tokenizer.from_file(self.model_path)
        decoded = tokenizer.decode(token_ids)
        return decoded

if __name__ == "__main__":
    # Example usage
    tokenizer = myTokenizer(default_config())
    tokenizer.train_tokenizer()
    text = "Hello, how are you?"
    token_ids = tokenizer.encode(text)
    print("Encoded:", token_ids)
    decoded_text = tokenizer.decode(token_ids.ids)
    print("Decoded:", decoded_text)