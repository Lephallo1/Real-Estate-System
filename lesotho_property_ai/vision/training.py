"""Training utilities for the vision side of the property-marketing assignment.

The project now uses two related vision stories:
1. a house-only multi-task CNN for bedrooms / condition / style / environment
2. a residential property-type classifier for House vs Townhouse vs Apartment
3. a focused bedroom-only classifier to strengthen the weakest Module 2 target

Keeping both in one module lets us reuse the same image-loading, training, and
artifact-writing logic while still producing separate saved results.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DEFAULT_TASK_COLUMNS = (
    "cnn_bedroom_class",
    "environment",
    "style",
    "condition",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _coarse_bedroom_class(value: object) -> str:
    """Bucket sparse bedroom labels into more learnable groups."""

    label = str(value).strip()
    if label in {"1", "2"}:
        return "1-2"
    if label == "3":
        return "3"
    if label in {"4", "5+", "5"}:
        return "4+"
    try:
        numeric = int(float(label))
    except (TypeError, ValueError):
        return label or "unknown"
    if numeric <= 2:
        return "1-2"
    if numeric == 3:
        return "3"
    return "4+"


@dataclass(slots=True)
class VisionTrainingResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    artifact_paths: dict[str, str]


class HouseVisionTrainer:
    """Train either the house-only multi-task model or a focused auxiliary vision model."""

    def __init__(
        self,
        task_columns: tuple[str, ...] = DEFAULT_TASK_COLUMNS,
        target_property_type: str | None = "House",
        random_state: int = 42,
        artifact_prefix: str = "house_vision",
    ) -> None:
        self.task_columns = task_columns
        self.target_property_type = target_property_type
        self.random_state = random_state
        self.artifact_prefix = artifact_prefix
        self.torch_available = self._check_torch()

    @staticmethod
    def _check_torch() -> bool:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def train(
        self,
        image_dataset: pd.DataFrame,
        output_dir: str | Path,
        epochs: int = 4,
        batch_size: int = 16,
        use_pretrained: bool = True,
    ) -> VisionTrainingResult:
        """Train the configured vision model and save reusable artifacts."""

        dataset = self._prepare_dataset(image_dataset)
        if dataset.empty:
            raise RuntimeError("No usable image rows were found for vision training.")

        task_columns = self._select_task_columns(dataset)
        if not task_columns:
            raise RuntimeError("No trainable house-vision tasks were found in the dataset.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.torch_available:
            try:
                return self._train_with_torch(
                    dataset=dataset,
                    task_columns=task_columns,
                    output_dir=output_path,
                    epochs=epochs,
                    batch_size=batch_size,
                    use_pretrained=use_pretrained,
                )
            except Exception:
                return self._train_with_fallback(dataset, task_columns, output_path)

        return self._train_with_fallback(dataset, task_columns, output_path)

    def _prepare_dataset(self, image_dataset: pd.DataFrame) -> pd.DataFrame:
        """Resolve image paths and apply any task-specific property-type filter."""

        df = image_dataset.copy()
        df["image_path"] = df["image_path"].map(self._resolve_image_path)
        # The house-only model keeps a strict filter here; the property-type model
        # disables it so it can learn across the residential categories.
        if self.target_property_type is not None:
            df = df[df["cnn_property_type"] == self.target_property_type].copy()
        df = df[df["split"].isin(["train", "val", "test"])].copy()
        df = df[df["image_path"].map(lambda value: Path(value).exists())].copy()
        return df.reset_index(drop=True)

    def _select_task_columns(self, dataset: pd.DataFrame) -> list[str]:
        """Keep only tasks that have enough label coverage to train safely."""

        train_df = dataset[dataset["split"] == "train"]
        selected: list[str] = []
        for column in self.task_columns:
            if column not in train_df:
                continue
            counts = train_df[column].dropna().astype(str).value_counts()
            if len(counts) < 2:







                continue
            if int(counts.min()) < 2:
                continue
            all_labels = set(dataset[column].dropna().astype(str).unique().tolist())
            train_labels = set(counts.index.astype(str).tolist())
            if not all_labels.issubset(train_labels):
                continue
            selected.append(column)
        return selected

    def _train_with_fallback(
        self,
        dataset: pd.DataFrame,
        task_columns: list[str],
        output_dir: Path,
    ) -> VisionTrainingResult:
        feature_rows = [self._extract_image_features(Path(path)) for path in dataset["image_path"]]
        features = pd.DataFrame(feature_rows, index=dataset.index)

        split_frames = {
            split: dataset[dataset["split"] == split].copy()
            for split in ("train", "val", "test")
        }
        feature_splits = {
            split: features.loc[frame.index].to_numpy(dtype=float)
            for split, frame in split_frames.items()
        }

        predictions = dataset[
            [
                "property_id",
                "source",
                "image_index",
                "image_path",
                "district_canonical",
                "locality",
                "split",
            ]
        ].copy()
        metrics: dict[str, Any] = {
            "mode": "fallback_image_classifier",
            "target_property_type": self.target_property_type or "all_residential_types",
            "tasks": {},
            "rows": int(len(dataset)),
        }
        trained_models: dict[str, Any] = {}
        label_maps: dict[str, list[str]] = {}

        for task in task_columns:
            label_values = sorted(split_frames["train"][task].astype(str).unique().tolist())
            label_to_index = {label: index for index, label in enumerate(label_values)}
            label_maps[task] = label_values

            y_train = split_frames["train"][task].astype(str).map(label_to_index).to_numpy()
            model = RandomForestClassifier(
                n_estimators=250,
                random_state=self.random_state,
                class_weight="balanced_subsample",
            )
            model.fit(feature_splits["train"], y_train)
            trained_models[task] = model

            metrics["tasks"][task] = {}
            for split, frame in split_frames.items():
                if frame.empty:
                    metrics["tasks"][task][split] = {"image_accuracy": None, "property_accuracy": None}
                    continue
                y_true = frame[task].astype(str).tolist()
                y_pred_indices = model.predict(feature_splits[split])
                y_pred = [label_values[int(index)] for index in y_pred_indices]
                predictions.loc[frame.index, f"actual_{task}"] = y_true
                predictions.loc[frame.index, f"predicted_{task}"] = y_pred
                image_accuracy = accuracy_score(y_true, y_pred)
                property_accuracy = self._property_level_accuracy(frame, task, y_pred)
                metrics["tasks"][task][split] = {
                    "image_accuracy": round(float(image_accuracy), 3),
                    "property_accuracy": round(float(property_accuracy), 3),
                    "rows": int(len(frame)),
                    "labels": label_values,
                }

        predictions_path = output_dir / f"{self.artifact_prefix}_predictions.csv"
        metrics_path = output_dir / f"{self.artifact_prefix}_metrics.json"
        model_path = output_dir / f"{self.artifact_prefix}_fallback_models.pkl"
        label_map_path = output_dir / f"{self.artifact_prefix}_label_maps.json"

        predictions.to_csv(predictions_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with model_path.open("wb") as handle:
            pickle.dump({"models": trained_models, "label_maps": label_maps}, handle)
        label_map_path.write_text(json.dumps(label_maps, indent=2), encoding="utf-8")

        return VisionTrainingResult(
            metrics=metrics,
            predictions=predictions,
            artifact_paths={
                "predictions_csv": str(predictions_path),
                "metrics_json": str(metrics_path),
                "model_pickle": str(model_path),
                "label_maps_json": str(label_map_path),
            },
        )

    def _train_with_torch(
        self,
        dataset: pd.DataFrame,
        task_columns: list[str],
        output_dir: Path,
        epochs: int,
        batch_size: int,
        use_pretrained: bool,
    ) -> VisionTrainingResult:
        import copy

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        from torchvision import models, transforms

        label_maps = {
            task: sorted(dataset.loc[dataset["split"] == "train", task].astype(str).unique().tolist())
            for task in task_columns
        }
        label_to_index = {
            task: {label: index for index, label in enumerate(labels)}
            for task, labels in label_maps.items()
        }

        class ImageDataset(Dataset):
            def __init__(self, frame: pd.DataFrame, transform) -> None:
                self.frame = frame.reset_index(drop=True)
                self.transform = transform

            def __len__(self) -> int:
                return len(self.frame)

            def __getitem__(self, index: int):
                row = self.frame.iloc[index]
                with Image.open(row["image_path"]) as source_image:
                    image = source_image.convert("RGB")
                tensor = self.transform(image)
                targets = {
                    task: torch.tensor(label_to_index[task][str(row[task])], dtype=torch.long)
                    for task in task_columns
                }
                metadata = {
                    "property_id": row["property_id"],
                    "image_path": row["image_path"],
                    "split": row["split"],
                }
                return tensor, targets, metadata

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(192, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
                transforms.RandomRotation(8),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((208, 208)),
                transforms.CenterCrop(192),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

        split_frames = {
            split: dataset[dataset["split"] == split].copy()
            for split in ("train", "val", "test")
        }
        train_dataset = ImageDataset(split_frames["train"], train_transform)
        train_weight_components: list[np.ndarray] = []
        task_loss_importance = {
            "cnn_bedroom_class": 1.4,
            "environment": 1.0,
            "style": 1.0,
            "condition": 0.8,
        }
        class_weights_map: dict[str, torch.Tensor] = {}
        for task in task_columns:
            counts = split_frames["train"][task].astype(str).value_counts()
            total = float(counts.sum())
            weights = []
            sample_weights = []
            for label in label_maps[task]:
                count = max(int(counts.get(label, 1)), 1)
                weights.append(total / (len(label_maps[task]) * count))
            for label in split_frames["train"][task].astype(str).tolist():
                sample_weights.append(total / (len(label_maps[task]) * max(int(counts.get(label, 1)), 1)))
            class_weights_map[task] = torch.tensor(weights, dtype=torch.float32)
            train_weight_components.append(np.asarray(sample_weights, dtype=np.float32) * task_loss_importance.get(task, 1.0))

        if train_weight_components:
            train_sample_weights = np.mean(np.vstack(train_weight_components), axis=0)
        else:
            train_sample_weights = np.ones(len(split_frames["train"]), dtype=np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
            num_samples=len(train_sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        eval_loaders = {
            split: DataLoader(ImageDataset(frame, eval_transform), batch_size=batch_size, shuffle=False)
            for split, frame in split_frames.items()
        }

        weights = None
        if use_pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except AttributeError:
                weights = None
        try:
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(weights=None)
            weights = None

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        class MultiHeadVisionModel(nn.Module):
            def __init__(self, feature_extractor, hidden_dim: int, head_sizes: dict[str, int]) -> None:
                super().__init__()
                self.feature_extractor = feature_extractor
                self.neck = nn.Sequential(
                    nn.Linear(hidden_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.35),
                )
                self.heads = nn.ModuleDict(
                    {
                        task: nn.Linear(256, output_size)
                        for task, output_size in head_sizes.items()
                    }
                )

            def forward(self, inputs):
                features = self.feature_extractor(inputs)
                features = self.neck(features)
                return {task: head(features) for task, head in self.heads.items()}

        model = MultiHeadVisionModel(backbone, feature_dim, {task: len(labels) for task, labels in label_maps.items()})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if weights is not None:
            for parameter in model.feature_extractor.parameters():
                parameter.requires_grad = False
            for parameter in model.feature_extractor.layer3.parameters():
                parameter.requires_grad = True
            for parameter in model.feature_extractor.layer4.parameters():
                parameter.requires_grad = True
            trainable_parameters = [
                {
                    "params": [parameter for parameter in model.feature_extractor.layer3.parameters() if parameter.requires_grad],
                    "lr": 2e-4,
                },
                {
                    "params": [parameter for parameter in model.feature_extractor.layer4.parameters() if parameter.requires_grad],
                    "lr": 3e-4,
                },
                {
                    "params": list(model.neck.parameters()) + list(model.heads.parameters()),
                    "lr": 1e-3,
                },
            ]
        else:
            trainable_parameters = [{"params": model.parameters(), "lr": 3e-4}]

        optimizer = torch.optim.AdamW(trainable_parameters, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
        )
        criteria = {
            task: nn.CrossEntropyLoss(weight=class_weights_map[task].to(device))
            for task in task_columns
        }

        def evaluate_split(split: str) -> tuple[dict[str, list[str]], dict[str, Any], float]:
            loader = eval_loaders[split]
            frame = split_frames[split]
            task_predictions: dict[str, list[str]] = {task: [] for task in task_columns}
            split_metrics: dict[str, Any] = {}
            if frame.empty:
                return task_predictions, split_metrics, 0.0

            model.eval()
            with torch.no_grad():
                for images, _, _ in loader:
                    images = images.to(device)
                    outputs = model(images)
                    for task in task_columns:
                        pred_indices = outputs[task].argmax(dim=1).cpu().tolist()
                        task_predictions[task].extend([label_maps[task][int(index)] for index in pred_indices])

            score_parts: list[float] = []
            for task in task_columns:
                y_true = frame[task].astype(str).tolist()
                y_pred = task_predictions[task]
                image_accuracy = accuracy_score(y_true, y_pred)
                property_accuracy = self._property_level_accuracy(frame, task, y_pred)
                split_metrics[task] = {
                    "image_accuracy": round(float(image_accuracy), 3),
                    "property_accuracy": round(float(property_accuracy), 3),
                    "rows": int(len(frame)),
                    "labels": label_maps[task],
                }
                weighted_score = property_accuracy * task_loss_importance.get(task, 1.0)
                score_parts.append(weighted_score)
            selection_score = float(sum(score_parts) / max(sum(task_loss_importance.get(task, 1.0) for task in task_columns), 1.0))
            return task_predictions, split_metrics, selection_score

        history: list[dict[str, Any]] = []
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_score = float("-inf")
        patience = 0

        for epoch in range(1, max(1, epochs) + 1):
            model.train()
            running_loss = 0.0
            batch_count = 0
            for images, targets, _ in train_loader:
                images = images.to(device)
                target_tensors = {task: tensor.to(device) for task, tensor in targets.items()}
                optimizer.zero_grad()
                outputs = model(images)
                loss = sum(
                    criteria[task](outputs[task], target_tensors[task]) * task_loss_importance.get(task, 1.0)
                    for task in task_columns
                )
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                batch_count += 1

            _, val_metrics_by_task, selection_score = evaluate_split("val")
            scheduler.step(selection_score)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(running_loss / max(batch_count, 1), 4),
                    "val_selection_score": round(selection_score, 4),
                }
            )
            if selection_score > best_score:
                best_score = selection_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= 2 and epoch >= 4:
                break

        model.load_state_dict(best_state)
        predictions = dataset[
            [
                "property_id",
                "source",
                "image_index",
                "image_path",
                "district_canonical",
                "locality",
                "split",
            ]
        ].copy()
        metrics: dict[str, Any] = {
            "mode": "torch_multitask_cnn",
            "target_property_type": self.target_property_type or "all_residential_types",
            "tasks": {},
            "rows": int(len(dataset)),
            "epochs": int(max(1, epochs)),
            "weights": "pretrained" if weights is not None else "scratch",
            "best_epoch": int(best_epoch),
            "selection_metric": round(float(best_score), 4) if best_score != float("-inf") else None,
            "history": history,
        }

        for split in ("train", "val", "test"):
            frame = split_frames[split]
            if frame.empty:
                continue
            all_predictions, task_metrics, _ = evaluate_split(split)
            for task in task_columns:
                predictions.loc[frame.index, f"actual_{task}"] = frame[task].astype(str).tolist()
                predictions.loc[frame.index, f"predicted_{task}"] = all_predictions[task]
                metrics["tasks"].setdefault(task, {})
                metrics["tasks"][task][split] = task_metrics[task]

        predictions_path = output_dir / f"{self.artifact_prefix}_predictions.csv"
        metrics_path = output_dir / f"{self.artifact_prefix}_metrics.json"
        model_path = output_dir / f"{self.artifact_prefix}_multitask.pt"
        label_map_path = output_dir / f"{self.artifact_prefix}_label_maps.json"

        predictions.to_csv(predictions_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_maps": label_maps,
                "tasks": task_columns,
                "target_property_type": self.target_property_type or "all_residential_types",
            },
            model_path,
        )
        label_map_path.write_text(json.dumps(label_maps, indent=2), encoding="utf-8")

        return VisionTrainingResult(
            metrics=metrics,
            predictions=predictions,
            artifact_paths={
                "predictions_csv": str(predictions_path),
                "metrics_json": str(metrics_path),
                "model_path": str(model_path),
                "label_maps_json": str(label_map_path),
            },
        )

    @staticmethod
    def _extract_image_features(image_path: Path) -> dict[str, float]:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            resized = image.resize((96, 96))
            pixels = np.asarray(resized, dtype=np.float32) / 255.0
        channel_means = pixels.mean(axis=(0, 1))
        channel_stds = pixels.std(axis=(0, 1))
        grayscale = pixels.mean(axis=2)
        brightness = float(grayscale.mean())
        contrast = float(grayscale.std())
        return {
            "red_mean": float(channel_means[0]),
            "green_mean": float(channel_means[1]),
            "blue_mean": float(channel_means[2]),
            "red_std": float(channel_stds[0]),
            "green_std": float(channel_stds[1]),
            "blue_std": float(channel_stds[2]),
            "brightness": brightness,
            "contrast": contrast,
            "width_height_ratio": round(resized.width / max(resized.height, 1), 4),
            "pixel_energy": float(np.square(grayscale).mean()),
        }

    @staticmethod
    def _property_level_accuracy(frame: pd.DataFrame, task: str, image_predictions: list[str]) -> float:
        working = frame[["property_id", task]].copy()
        working["predicted"] = image_predictions
        grouped = working.groupby("property_id", sort=False)
        total = 0
        correct = 0
        for _, group in grouped:
            actual = str(group[task].iloc[0])
            predicted = Counter(group["predicted"].tolist()).most_common(1)[0][0]
            total += 1
            correct += int(actual == predicted)
        return correct / max(total, 1)

    @staticmethod
    def _resolve_image_path(value: object) -> str:
        original = str(value).strip()
        if not original:
            return original
        original_path = Path(original)
        if original_path.exists():
            return str(original_path)

        normalized = original.replace("/", "\\")
        marker = "generated\\images\\"
        lowered = normalized.lower()
        if marker in lowered:
            suffix = normalized[lowered.index(marker) :].replace("\\", "/")
            candidate = PROJECT_ROOT / Path(suffix)
            if candidate.exists():
                return str(candidate)
        return original


def train_house_vision_model(
    image_dataset_csv: str | Path,
    output_dir: str | Path,
    epochs: int = 4,
    batch_size: int = 16,
    use_pretrained: bool = True,
) -> VisionTrainingResult:
    dataframe = pd.read_csv(Path(image_dataset_csv))
    trainer = HouseVisionTrainer()
    return trainer.train(
        image_dataset=dataframe,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        use_pretrained=use_pretrained,
    )


def train_residential_property_type_model(
    image_dataset_csv: str | Path,
    output_dir: str | Path,
    epochs: int = 4,
    batch_size: int = 16,
    use_pretrained: bool = True,
) -> VisionTrainingResult:
    """Train the missing Module 2 classifier for residential property type.

    This model uses the broader residential image dataset rather than the
    house-only reviewed dataset so it can classify the categories we still have
    enough data for: House, Townhouse, and Apartment.
    """

    dataframe = pd.read_csv(Path(image_dataset_csv))
    working = dataframe.copy()
    supported_types = {"House", "Townhouse", "Apartment"}
    if "cnn_property_type" in working.columns:
        working = working[working["cnn_property_type"].astype(str).isin(supported_types)].copy()
    trainer = HouseVisionTrainer(
        task_columns=("cnn_property_type",),
        target_property_type=None,
        artifact_prefix="residential_property_type",
    )
    return trainer.train(
        image_dataset=working,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        use_pretrained=use_pretrained,
    )


def train_house_bedroom_model(
    image_dataset_csv: str | Path,
    output_dir: str | Path,
    epochs: int = 6,
    batch_size: int = 16,
    use_pretrained: bool = True,
) -> VisionTrainingResult:
    """Train a dedicated house-bedroom classifier on the reviewed house dataset.

    The main multi-task model is kept for the broader story, but bedroom count is
    the weakest task. This specialist model gives that target its own optimization
    path instead of making it compete with style, condition, and environment.
    It also uses coarser bedroom groups so the rare `1` and `5+` labels do not
    destabilize the model.
    """

    dataframe = pd.read_csv(Path(image_dataset_csv))
    dataframe = dataframe.copy()
    dataframe["cnn_bedroom_class"] = dataframe["cnn_bedroom_class"].map(_coarse_bedroom_class)
    trainer = HouseVisionTrainer(
        task_columns=("cnn_bedroom_class",),
        target_property_type="House",
        artifact_prefix="house_bedroom",
    )
    result = trainer.train(
        image_dataset=dataframe,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        use_pretrained=use_pretrained,
    )
    result.metrics["bedroom_label_scheme"] = "coarse_groups_1-2_3_4+"
    return result
