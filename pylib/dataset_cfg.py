#!/usr/bin/env python3

import dataclasses
import json
import os
from typing import NoReturn, Self


@dataclasses.dataclass
class DatasetCfg(object):
	key: str
	ref: str = None
	color: str = "#808080"
	display_name: str = None
	technology: str = "Unknown"
	n_samples: int = 0

	def __post_init__(self) -> NoReturn:
		if self.display_name is None:
			self.display_name = self.key
		return


class DatasetCfgLib(dict[DatasetCfg]):
	@classmethod
	def from_json(cls, fname: str = None) -> Self:
		if fname is None:
			fname = os.path.join(os.path.dirname(__file__),
				"data/dataset_cfg.json")
		with open(fname, "r") as f:
			data: dict[str] = json.load(f)

		return cls((k, DatasetCfg(key=k, **v)) for k, v in data.items())

	def __getitem__(self, key: str) -> DatasetCfg:
		cfg = super().__getitem__(key)
		if cfg.ref is not None:
			cfg = self[cfg.ref]
		return cfg

	def __missing__(self, key: str) -> DatasetCfg:
		self[key] = ret = DatasetCfg(key=key)
		return ret

	def add_cfg(self, cfg: DatasetCfg) -> NoReturn:
		self[cfg.key] = cfg
		return
