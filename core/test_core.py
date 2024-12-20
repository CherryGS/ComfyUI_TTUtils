import pytest as pt


class TestParam:
    def test_schema(self):
        from .core import Int, Custom

        assert (
            Int("t").schema()
            == "('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 64, 'display': 'number'})"
        )

        assert Custom("t", pt).schema() == "('pytest',)"
