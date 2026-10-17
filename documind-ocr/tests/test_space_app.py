import importlib.util
import os

import pytest

gr = pytest.importorskip("gradio")
pytest.importorskip("pytesseract")
import shutil  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract missing")


def test_space_extract_review_export_roundtrip():
    spec = importlib.util.spec_from_file_location(
        "space_app", os.path.join(os.path.dirname(__file__), "..", "space", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = mod.sample()
    preview, rows, msg = mod.extract(path)
    assert os.path.exists(preview) and {r[0] for r in rows} == {"vendor_name", "invoice_date", "total_amount"}
    assert "**" in msg and len(mod.queue()) >= 0
    assert os.path.exists(mod.export())
    with pytest.raises(gr.Error):
        mod.extract(None)
    assert mod.build() is not None
