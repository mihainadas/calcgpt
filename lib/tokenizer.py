"""
CalcGPT Tokenizer

Supports both character-level and number-level tokenization for arithmetic expressions.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Literal


# Special tokens for padding and end-of-sequence
PAD_TOKEN = '<pad>'
EOS_TOKEN = '<eos>'

# Tokenization modes
TokenizationMode = Literal['char', 'number']


class CalcGPTTokenizer:
    """Tokenizer for arithmetic expressions with character or number-level modes"""
    
    def __init__(self, examples: List[str], mode: TokenizationMode = 'char'):
        """Initialize tokenizer from examples
        
        Args:
            examples: List of training examples
            mode: 'char' for character-level, 'number' for number-level (0-99)
        """
        if not examples:
            raise ValueError("Examples cannot be empty")
        
        self.mode = mode
        
        if mode == 'char':
            self._build_char_vocab(examples)
        elif mode == 'number':
            self._build_number_vocab(examples)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'char' or 'number'")
        
        # Calculate max length based on mode
        self.maxlen = max(len(self.encode(example)) for example in examples)
    
    def _build_char_vocab(self, examples: List[str]):
        """Build character-level vocabulary"""
        # Create vocabulary from all characters in examples
        chars = sorted(set(''.join(examples)))
        tokens = [PAD_TOKEN, EOS_TOKEN] + chars
        
        self.vocab = {token: i for i, token in enumerate(tokens)}
        self.id2char = {i: token for token, i in self.vocab.items()}
    
    def _build_number_vocab(self, examples: List[str]):
        """Build number-level vocabulary (numbers 0-99 + operators)"""
        # Start with special tokens
        tokens = [PAD_TOKEN, EOS_TOKEN]
        
        # Add all numbers from 0 to 99
        for i in range(100):
            tokens.append(str(i))
        
        # Add operators and other non-numeric characters found in examples
        all_chars = set(''.join(examples))
        operators = set()
        for char in all_chars:
            if not char.isdigit():  # Non-digit characters (operators, =, spaces, etc.)
                operators.add(char)
        
        tokens.extend(sorted(operators))
        
        self.vocab = {token: i for i, token in enumerate(tokens)}
        self.id2char = {i: token for token, i in self.vocab.items()}
    
    @classmethod
    def from_dataset(cls, dataset_path: Path = None, mode: TokenizationMode = 'char') -> 'CalcGPTTokenizer':
        """Create tokenizer from dataset file
        
        Args:
            dataset_path: Path to dataset file (defaults to standard location)
            mode: 'char' for character-level, 'number' for number-level
            
        Returns:
            CalcGPTTokenizer instance
        """
        if dataset_path is None:
            dataset_path = Path('datasets/ds-calcgpt.txt')
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            examples = [line.strip() for line in f if line.strip()]
        
        if not examples:
            raise ValueError("Dataset is empty")
        
        return cls(examples, mode)
    
    def _parse_tokens(self, text: str) -> List[str]:
        """Parse text into tokens based on mode"""
        if self.mode == 'char':
            return list(text)
        elif self.mode == 'number':
            # Use regex to split into numbers and non-numbers
            # This will separate "12+34=46" into ["12", "+", "34", "=", "46"]
            pattern = r'(\d+|[^\d])'
            tokens = re.findall(pattern, text)
            return [token for token in tokens if token]  # Remove empty strings
    
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """Encode text to token IDs
        
        Args:
            text: String to encode
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        tokens = []
        parsed_tokens = self._parse_tokens(text)
        
        for token in parsed_tokens:
            if token in self.vocab:
                tokens.append(self.vocab[token])
            # Skip unknown tokens silently
        
        if add_eos:
            tokens.append(self.vocab[EOS_TOKEN])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded string (skips special tokens)
        """
        chars = []
        special_ids = {self.vocab[PAD_TOKEN], self.vocab[EOS_TOKEN]}
        
        for token_id in token_ids:
            if token_id in self.id2char and token_id not in special_ids:
                chars.append(self.id2char[token_id])
        
        return ''.join(chars)
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size"""
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID"""
        return self.vocab[PAD_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token ID"""
        return self.vocab[EOS_TOKEN]
    
    @property
    def max_length(self) -> int:
        """Maximum sequence length"""
        return self.maxlen



