#!/usr/bin/env python3

import functools
import logging


class Logger(logging.Logger):
	@functools.wraps(logging.Logger.__init__)
	def __init__(self, *ka, **kw):
		super().__init__(*ka, **kw)
		formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
		handler = logging.StreamHandler()
		handler.setFormatter(formatter)
		self.addHandler(handler)
		return
