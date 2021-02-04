import pytest
from dorc.spec.docstring import GoogleDocstring


@pytest.mark.spec
def test_docstring_should_parse_multiline_description():
    doc_str = """Load model weights or trainer state from a given filename. The file must be
        present in the `savedir`.

        Args:
            weights: The name of the weights file
            method: How to load the saves

        Returns:
            An instance of :class:`Return`

        Not sure right now, when something should be allowed to load. If it's
        paused? In the middle of current session? Should the session be
        restarted?

        """
    doc = GoogleDocstring(doc_str)
    assert hasattr(doc, "description")
    assert "load model weights" in doc.description.lower()
    assert "present in the `savedir`" in doc.description.lower()


@pytest.mark.spec
def test_docstring_parse_should_not_parse_empty_description():
    doc_str = """
        Args:
            weights: The name of the weights file
            method: How to load the saves

        Returns:
            An instance of :class:`Return`

        Not sure right now, when something should be allowed to load. If it's
        paused? In the middle of current session? Should the session be
        restarted?

        """
    doc = GoogleDocstring(doc_str)
    assert not hasattr(doc, "description")
