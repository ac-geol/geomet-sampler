"""Exceptions raised by the library. The CLI turns these into readable messages."""

from __future__ import annotations


class GeometSamplerError(Exception):
    """Base class for every error this library raises deliberately."""


class MappingError(GeometSamplerError):
    """A configured column does not exist in the file it was mapped against.

    Raised once, listing every bad mapping across every source, so the user fixes the
    whole config in one pass rather than one column per run.
    """


class ValidationFailedError(GeometSamplerError):
    """Validation produced ERROR-severity issues and --force was not given."""


class PipelineError(GeometSamplerError):
    """A stage could not produce output at all."""
