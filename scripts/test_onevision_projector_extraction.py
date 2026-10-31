#!/usr/bin/env python3
"""Dependency-light checks for OneVision projector capture and analysis inclusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from structured_color_utils import LlavaHelper


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    @property
    def shape(self):
        return self.values.shape

    def reshape(self, *shape):
        return FakeTensor(self.values.reshape(*shape))

    def mean(self, dim=0):
        return FakeTensor(self.values.mean(axis=dim))

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()

    def __getitem__(self, item):
        return FakeTensor(self.values[item])


class FakeHandle:
    def __init__(self, module, hook):
        self.module = module
        self.hook = hook

    def remove(self):
        self.module.hooks.remove(self.hook)


class FakeProjector:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle(self, hook)

    def __call__(self, values):
        output = FakeTensor(values.values[:, :2] * 2.0)
        for hook in list(self.hooks):
            hook(self, (values,), output)
        return output


class FakeTorch:
    @staticmethod
    def stack(values, dim=0):
        return FakeTensor(np.stack([value.values for value in values], axis=dim))


class FakeModel:
    def __init__(self, projector):
        self.model = type("Inner", (), {})()
        self.model.visual = type("Visual", (), {"merger": projector})()
        self.projector = projector

    def __call__(self, **_kwargs):
        projected = self.projector(FakeTensor([[1, 3, 5], [5, 7, 9]]))
        return {"projected": projected}


def main() -> None:
    projector = FakeProjector()
    helper = object.__new__(LlavaHelper)
    helper.model = FakeModel(projector)
    helper.torch = FakeTorch()
    helper.projector = helper._find_projector()

    outputs, pre, post = helper._forward_with_projector_capture({}, "mean")
    np.testing.assert_allclose(pre.numpy(), [3, 5, 7])
    np.testing.assert_allclose(post.numpy(), [6, 10])
    assert "projected" in outputs
    assert not projector.hooks, "The temporary hook must always be removed"

    class FailingModel:
        def __call__(self, **kwargs):
            raise RuntimeError("simulated failure")
    helper.model = FailingModel()
    try:
        helper._forward_with_projector_capture({}, "mean")
    except RuntimeError:
        pass
    assert not projector.hooks, "Forward failures must also remove hooks"
    print("Projector capture and failure cleanup passed; CSV inclusion is tested in test_alignment_v2.py.")


if __name__ == "__main__":
    main()
