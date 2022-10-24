# Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import ABC
from dataclasses import dataclass

# Define objects to represent the amortization type metadata and config
class BaseAmortization(ABC):
    finetune_during_training: bool

@dataclass
class NoAmortization(BaseAmortization):
    finetune_during_training: bool = True # Needs to be true


@dataclass
class RegressionAmortization(BaseAmortization):
    finetune_during_training: bool = True # Needs to be true


@dataclass
class ObjectiveAmortization(BaseAmortization):
    finetune_during_training: bool = True


@dataclass
class W2GNAmortization(BaseAmortization):
    finetune_during_training: bool = False
    cycle_loss_weight: float = None
    regularize_D: bool = True
    only_amortize_H: bool = False
