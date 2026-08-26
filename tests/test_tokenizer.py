"""Standard-library tests for tokenizer behavior and artifact stability."""

import json
import tempfile
import unittest
from pathlib import Path

from lib.tokenizer import TOKENIZER_FILENAME, CalcGPTTokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = ["1+2=3", "3-1=2", "9+9=18"]

    def test_character_round_trip(self) -> None:
        tokenizer = CalcGPTTokenizer(self.examples)
        encoded = tokenizer.encode("1+2=3")
        self.assertEqual(tokenizer.decode(encoded), "1+2=3")

    def test_unknown_character_is_rejected_without_partial_encoding(self) -> None:
        tokenizer = CalcGPTTokenizer(self.examples)
        with self.assertRaisesRegex(ValueError, r"Unknown token '\*'.*position 1"):
            tokenizer.encode("1*2")

    def test_unknown_number_token_is_rejected(self) -> None:
        tokenizer = CalcGPTTokenizer(self.examples, mode="number")
        with self.assertRaisesRegex(ValueError, "Unknown token '100'"):
            tokenizer.encode("100+1")

    def test_unknown_token_id_is_rejected(self) -> None:
        tokenizer = CalcGPTTokenizer(self.examples)
        with self.assertRaisesRegex(ValueError, "Unknown token ID 999"):
            tokenizer.decode([999])

    def test_artifact_round_trip_and_deterministic_bytes(self) -> None:
        tokenizer = CalcGPTTokenizer(self.examples)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = tokenizer.save_pretrained(root / "first")
            second = tokenizer.save_pretrained(root / "second")
            self.assertEqual(first.name, TOKENIZER_FILENAME)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            loaded = CalcGPTTokenizer.from_pretrained(first.parent)
            self.assertEqual(loaded.mode, tokenizer.mode)
            self.assertEqual(loaded.vocab, tokenizer.vocab)
            self.assertEqual(loaded.max_length, tokenizer.max_length)
            self.assertEqual(loaded.encode("3-1=2"), tokenizer.encode("3-1=2"))

    def test_rejects_incompatible_artifact_version(self) -> None:
        tokenizer = CalcGPTTokenizer(self.examples)
        with tempfile.TemporaryDirectory() as directory:
            artifact = tokenizer.save_pretrained(directory)
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload["format_version"] = 999
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported format version 999"):
                CalcGPTTokenizer.from_pretrained(directory)

    def test_rejects_special_token_metadata_mismatch(self) -> None:
        tokenizer = CalcGPTTokenizer(self.examples)
        with tempfile.TemporaryDirectory() as directory:
            artifact = tokenizer.save_pretrained(directory)
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload["special_tokens"]["pad_token_id"] = 1
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "special_tokens metadata"):
                CalcGPTTokenizer.from_pretrained(artifact)


if __name__ == "__main__":
    unittest.main()
