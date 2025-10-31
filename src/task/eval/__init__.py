from typing import Literal
from collections import defaultdict
import logging

from .processors import (
    get_last_number,
    last_boxed_only_string,
    get_last_option,
)

logger = logging.getLogger("evaluation")

class Evaluator:
    def process_pairs(self, labels, predictions) -> None:
        assert len(labels) == len(predictions)
        for label, prediction in zip(labels, predictions):
            self.process_single_pair(label, prediction)
    def process_single_pair(self, label, prediction) -> None:
        raise NotImplementedError
    def scores(self):
        raise NotImplementedError
    def aggregate_score(self, aggregate_method):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def is_empty(self) -> bool:
        raise NotImplementedError


class ExactEvaluator(Evaluator):

    def __init__(self, process_fn):
        super().__init__()
        self.process_fn = process_fn
        self.reset()

    def process_single_pair(self, label, prediction) -> None:
        label_processed = self.process_fn(label)
        prediction_processed = self.process_fn(prediction)
        self._scores.append(int(label_processed == prediction_processed))

    def scores(self) -> list[float]:
        assert not self.is_empty(), "Querying for scores while evaluator is empty."
        return self._scores

    def aggregate_score(self, aggregate_method: Literal["mean"]="mean") -> float:
        assert not self.is_empty(), "Querying for scores while evaluator is empty."
        assert aggregate_method == "mean", f"Unrecognized aggregation method `{aggregate_method}`"
        return sum(self._scores) / len(self._scores)
    
    def reset(self) -> None:
        self._scores = []

    def is_empty(self) -> bool:
        return len(self._scores) == 0

class ExactLastNumberEvaluator(ExactEvaluator):
    def __init__(self):
        super().__init__(process_fn=get_last_number)

class ExactBoxedAnswerEvaluator(ExactEvaluator):
    def __init__(self):
        super().__init__(process_fn=last_boxed_only_string)

class ExactLastOptionEvaluator(ExactEvaluator):
    def __init__(self):
        super().__init__(process_fn=get_last_option)

class Throughput(Evaluator):
    def __init__(self):
        super().__init__()
        self.reset()

    def process_single_pair(self, label, prediction) -> None:
        n_tokens = label
        time = prediction
        assert n_tokens > 0
        assert time > 0
        self.n_tokens += n_tokens
        self.time += time

    def scores(self) -> list[float]:
        assert not self.is_empty(), "Querying for scores while evaluator is empty."
        return [self.aggregate_score()]

    def aggregate_score(self, aggregate_method: Literal["mean"]="mean") -> float:
        assert not self.is_empty(), "Querying for scores while evaluator is empty."
        assert aggregate_method == "mean", f"Unrecognized aggregation method `{aggregate_method}`"
        return self.n_tokens / self.time
    
    def reset(self) -> None:
        self.n_tokens = 0
        self.time = 0

    def is_empty(self) -> bool:
        return self.time == 0

class AcceptationRate(Evaluator):
    def __init__(self):
        super().__init__()
        self.reset()

    def process_single_pair(self, label, prediction) -> None:
        window_size = label
        n_accepted = prediction
        self._scores.extend([n / window_size for n in n_accepted])

    def scores(self) -> list[float]:
        assert not self.is_empty(), "Querying for scores while evaluator is empty."
        return self._scores

    def aggregate_score(self, aggregate_method: Literal["mean"]="mean") -> float:
        assert not self.is_empty(), "Querying for scores while evaluator is empty."
        assert aggregate_method == "mean", f"Unrecognized aggregation method `{aggregate_method}`"
        return sum(self._scores) / len(self._scores)
    
    def reset(self) -> None:
        self._scores = []

    def is_empty(self) -> bool:
        return len(self._scores) == 0

class AcceptanceLength(Evaluator):
    def __init__(self):
        super().__init__()
        self.reset()

    def process_single_pair(self, label, prediction) -> None:
        window_size = label
        n_accepted = prediction
        self._scores.extend(n_accepted)

    def scores(self) -> list[float]:
        assert not self.is_empty(), "Querying for scores while evaluator is empty."
        return self._scores

    def aggregate_score(self, aggregate_method: Literal["mean"]="mean") -> float:
        assert not self.is_empty(), "Querying for scores while evaluator is empty."
        assert aggregate_method == "mean", f"Unrecognized aggregation method `{aggregate_method}`"
        return sum(self._scores) / len(self._scores)
    
    def reset(self) -> None:
        self._scores = []

    def is_empty(self) -> bool:
        return len(self._scores) == 0

class AcceptanceLengthProportions(Evaluator):
    def __init__(self):
        super().__init__()
        self.reset()

    def process_single_pair(self, label, prediction) -> None:
        window_size = label
        n_accepted = prediction
        for n in n_accepted:
            self._counts[n] += 1
            self._total += 1

    def aggregate_score(self, aggregate_method: Literal["mean"]="mean") -> dict|float:
        assert not self.is_empty(), "Querying for scores while evaluator is empty."
        assert aggregate_method == "mean", f"Unrecognized aggregation method `{aggregate_method}`"
        return {
            k: self._counts[k] / self._total
            for k in sorted(self._counts.keys())
        }
    
    def reset(self) -> None:
        self._counts = defaultdict(lambda: 0)
        self._total = 0

    def is_empty(self) -> bool:
        return self._total == 0
