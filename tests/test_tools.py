import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"wapic_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_script("test")

    def test_exact_score(self):
        score = self.evaluator.score_sentences(
            [["B", "E"], ["S"]],
            [["B", "E"], ["S"]],
        )
        self.assertEqual(score, (100.0, 100.0, 100.0))

    def test_sentence_count_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "sentence count mismatch"):
            self.evaluator.score_sentences([["S"]], [])

    def test_empty_gold_fails(self):
        with self.assertRaisesRegex(ValueError, "gold contains no sentences"):
            self.evaluator.score_sentences([], [])

    def test_sentence_length_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            self.evaluator.score_sentences([["B", "E"]], [["S"]])


class DataToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare = load_script("prepare")
        cls.convert = load_script("convert")

    def test_words_to_bmes(self):
        self.assertEqual(
            self.prepare.words_to_bmes(["我", "中国"]),
            ["我 S", "中 B", "国 E"],
        )

    def test_character_classes(self):
        self.assertEqual(self.convert.classify(ord("中")), "H")
        self.assertEqual(self.convert.classify(ord("A")), "L")
        self.assertEqual(self.convert.classify(ord("１")), "D")
        self.assertEqual(self.convert.classify(ord("。")), "P")


@unittest.skipUnless(os.environ.get("WAPIC_BIN"), "WAPIC_BIN is not set")
class CliTests(unittest.TestCase):
    def test_missing_model_fails_cleanly(self):
        result = subprocess.run(
            [os.environ["WAPIC_BIN"], "-m", "/does/not/exist.wac"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Cannot open model", result.stderr)

    def test_more_than_eight_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.txt"
            model = Path(tmp) / "model.wac"
            train.write_text(
                "".join(f"{chr(0x4E00 + i)} L{i}\n\n" for i in range(9)),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    os.environ["WAPIC_BIN"],
                    "fit",
                    "-p",
                    str(ROOT / "data" / "pattern.txt"),
                    "-i",
                    "1",
                    "-t",
                    "1",
                    str(train),
                    str(model),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(model.is_file())
            self.assertGreater(model.stat().st_size, 0)


@unittest.skipUnless(os.environ.get("WAPIC_MODEL"), "WAPIC_MODEL is not set")
class PythonApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import wapic

        cls.segmenter = wapic.Segmenter(os.environ["WAPIC_MODEL"])

    def test_raw_apis_accept_spaces(self):
        text = "abc 123 中国"
        chars, tags = self.segmenter.tag(text)
        self.assertEqual("".join(chars), text)
        self.assertEqual(len(chars), len(tags))
        self.assertEqual(tags[3], "S")
        self.assertEqual(tags[7], "S")
        self.assertEqual("".join(self.segmenter.segment_raw(text)), text)

    def test_word_starts_follow_segment(self):
        self.assertEqual(
            self.segmenter.word_starts("abc 123 中国"),
            [0, 4, 8, 10],
        )

    def test_batch_matches_single_calls(self):
        texts = ["中国AI模型2.0", "abc 123 中国", ""] * 32
        self.assertEqual(
            self.segmenter.segment_batch(texts),
            [self.segmenter.segment(text) for text in texts],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("wapic_model"),
        "wapic-cws-model is not installed",
    )
    def test_default_model(self):
        import wapic
        import wapic_model

        self.assertEqual(
            wapic.Segmenter().segment("中华人民共和国"),
            wapic.Segmenter(wapic_model.model_path()).segment(
                "中华人民共和国"
            ),
        )


if __name__ == "__main__":
    unittest.main()
