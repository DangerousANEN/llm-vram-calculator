# -*- coding: utf-8 -*-
"""Typed errors for the config-grounded memory ledger.

Every refusal carries a stable machine-readable ``code`` so the CLI, the web UI
and the tests can agree on *why* an answer was refused instead of guessing.
"""
from __future__ import annotations


class CalcError(Exception):
    """Base class for all calculator refusals."""

    code = "error"
    exit_code = 2

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "details": self.details}


class InvalidParameter(CalcError):
    code = "invalid_parameter"


class UnknownModel(CalcError):
    code = "unknown_model"


class UnsupportedQuant(CalcError):
    code = "unsupported_quant"


class UnsupportedArchitecture(CalcError):
    """The config describes something this tool will not pretend to model."""

    code = "unsupported_architecture"


class UnsupportedFeature(CalcError):
    code = "unsupported_feature"


class BenchmarkError(CalcError):
    code = "benchmark_error"
    exit_code = 3
