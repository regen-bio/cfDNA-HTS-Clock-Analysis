#!/usr/bin/env python3

import dataclasses
import json
import os
from typing import NoReturn, Self, Sequence


@dataclasses.dataclass
class ClockCfg(object):
	key: str
	marker: str
	marker_char: str
	name: str = None
	display_name: str = None
	color: str = "#808080"

	def __post_init__(self) -> NoReturn:
		if self.name is None:
			self.name = self.key
		if self.display_name is None:
			self.display_name = self.key
		return


class ClockCfgLib(dict[ClockCfg]):
	@classmethod
	def from_json(cls, fname: str = None, *, select: Sequence[str] = None
	) -> Self:
		if fname is None:
			fname = os.path.join(os.path.dirname(__file__),
				"data/clock_cfg.json")
		with open(fname, "r") as f:
			data: dict[str, dict] = json.load(f)

		if select is not None:
			data = {k: v for k, v in data.items() if k in select}

		return cls((k, ClockCfg(key=k, **v)) for k, v in data.items())
