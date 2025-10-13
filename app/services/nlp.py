from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Tuple

import os
import re
import shutil
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from torch.utils.data import DataLoader
from transformers.utils import logging as hf_logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()


def _resolve_device() -> torch.device:
    """Choose CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class IntentPrediction:
    label: str
    confidence: float


class IntentRecognizer:
    """Tiny intent classifier powered by a BERT encoder."""

    MODEL_NAME: ClassVar[str] = "uer/albert-base-chinese-cluecorpussmall"
    INTENT_LABELS: ClassVar[Tuple[str, ...]] = (
        "查詢風險",
        "推薦餐廳",
        "回報雷點",
        "查詢公司風險",
    )
    INTENT_TO_ACTION: ClassVar[dict[str, str]] = {
        "查詢風險": "查詢餐廳",
        "推薦餐廳": "查詢餐廳",
        "回報雷點": "查詢餐廳",
        "查詢公司風險": "查詢公司",
    }

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.device = _resolve_device()
        self.cache_dir = cache_dir or Path("models/intent_classifier")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.cache_dir / "trained"
        model_ready = (model_path / "config.json").exists()
        tokenizer_source = model_path if model_ready else self.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if model_ready:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                num_labels=len(self.INTENT_LABELS),
            )
            self._train(model_path)

        self.model.to(self.device)
        self.model.eval()

    def _make_dataset(self) -> Dataset:
        """Build a miniature dataset for demonstration purposes."""
        samples = {
            "text": [
                "這家燒烤店安全嗎",
                "請推薦親子友善餐廳",
                "XX公司有違法紀錄嗎",
                "我要檢舉這間診所",
                "這家公司最近是不是出事了",
                "推薦環保餐廳",
                "我要回報這家很雷",
                "XX藥局安全嗎",
            ],
            "label": [0, 1, 3, 2, 3, 1, 2, 0],
        }
        dataset = Dataset.from_dict(samples)

        def tokenize(batch: dict[str, list[str]]):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=64,
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        return dataset

    def _train(self, output_dir: Path) -> None:
        dataset = self._make_dataset()
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)
        self.model.to(self.device)
        self.model.train()

        epochs = 12
        for _ in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = self.model(**batch, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.model.eval()

    def predict(self, text: str) -> IntentPrediction:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            scores = torch.softmax(logits, dim=-1)

        top_prob, top_idx = scores.max(dim=-1)
        label = self.INTENT_LABELS[top_idx.item()]
        return IntentPrediction(label=label, confidence=float(top_prob.item()))

    def predict_action(self, text: str) -> Tuple[str, IntentPrediction]:
        intent_pred = self.predict(text)
        action = self.INTENT_TO_ACTION.get(intent_pred.label, "查詢餐廳")
        return action, intent_pred


@dataclass
class EntityPrediction:
    text: str
    label: str
    confidence: float
    start: int
    end: int


class NamedEntityExtractor:
    """Wrapper around a BERT token classification model for Chinese NER."""

    MODEL_NAME: ClassVar[str] = "shibing624/bert4ner-base-chinese"
    KNOWN_ENTITY_MAP: ClassVar[dict[str, str]] = {
        "麥當勞": "店家",
        "肯德基": "店家",
        "星巴克": "店家",
        "全聯": "店家",
        "家樂福": "店家",
        "頂好": "店家",
        "康是美": "店家",
        "屈臣氏": "店家",
        "7-11": "店家",
        "7-ELEVEN": "店家",
        "全家": "店家",
    }
    ENTITY_SUFFIX_PATTERNS: ClassVar[List[Tuple[re.Pattern, str]]] = [
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}公司)"), "公司"),
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}企業)"), "公司"),
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}診所)"), "公司"),
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}牙醫)"), "公司"),
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}餐廳)"), "店家"),
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}燒烤店)"), "店家"),
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}火鍋店)"), "店家"),
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}旅館)"), "店家"),
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}飯店)"), "店家"),
        (re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,12}茶坊)"), "店家"),
    ]

    def __init__(self) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            task="token-classification",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            aggregation_strategy="simple",
            device=device,
        )

    @staticmethod
    def _classify_entity(word: str) -> str:
        if "公司" in word or "企業" in word or "股份" in word:
            return "公司"
        if "診所" in word or "藥局" in word or "醫院" in word:
            return "公司"
        return "店家"

    def extract(self, text: str) -> List[EntityPrediction]:
        ner_results = self.pipeline(text)
        predictions: List[EntityPrediction] = []
        for result in ner_results:
            entity_word = result["word"]
            label = result.get("entity_group", result.get("entity", "O"))
            if label == "O":
                continue
            predictions.append(
                EntityPrediction(
                    text=entity_word,
                    label=self._classify_entity(entity_word),
                    confidence=float(result["score"]),
                    start=int(result["start"]),
                    end=int(result["end"]),
                )
            )
        predictions = self._deduplicate(predictions)
        if not predictions:
            predictions = self._fallback_entities(text)
        return predictions

    @staticmethod
    def _deduplicate(predictions: List[EntityPrediction]) -> List[EntityPrediction]:
        unique: dict[tuple[int, int, str], EntityPrediction] = {}
        for pred in predictions:
            key = (pred.start, pred.end, pred.label)
            if key not in unique or unique[key].confidence < pred.confidence:
                unique[key] = pred
        return sorted(unique.values(), key=lambda item: item.start)

    def _fallback_entities(self, text: str) -> List[EntityPrediction]:
        fallback_predictions: list[EntityPrediction] = []

        for name, label in self.KNOWN_ENTITY_MAP.items():
            start = text.find(name)
            if start != -1:
                fallback_predictions.append(
                    EntityPrediction(
                        text=name,
                        label=label,
                        confidence=0.6,
                        start=start,
                        end=start + len(name),
                    )
                )

        for pattern, label in self.ENTITY_SUFFIX_PATTERNS:
            for match in pattern.finditer(text):
                entity_word = match.group(1)
                start = match.start(1)
                fallback_predictions.append(
                    EntityPrediction(
                        text=entity_word,
                        label=label,
                        confidence=0.4,
                        start=start,
                        end=start + len(entity_word),
                    )
                )

        return self._deduplicate(fallback_predictions)


class NLPOrchestrator:
    """Facade combining intent classification and NER."""

    def __init__(self) -> None:
        self.intent = IntentRecognizer()
        self.ner = NamedEntityExtractor()

    def analyze(self, text: str) -> tuple[str, IntentPrediction, List[EntityPrediction]]:
        action, intent_pred = self.intent.predict_action(text)
        entities = self.ner.extract(text)
        if action == "查詢公司" and all(ent.label != "公司" for ent in entities):
            # Ensure output remains consistent with the requested schema.
            entities = [
                EntityPrediction(
                    text=text,
                    label="公司",
                    confidence=0.0,
                    start=0,
                    end=len(text),
                )
            ]
        elif action == "查詢餐廳" and not entities:
            entities = [
                EntityPrediction(
                    text=text,
                    label="店家",
                    confidence=0.0,
                    start=0,
                    end=len(text),
                )
            ]
        return action, intent_pred, entities


# Provide a single global orchestrator instance that can be reused.
orchestrator = NLPOrchestrator()
