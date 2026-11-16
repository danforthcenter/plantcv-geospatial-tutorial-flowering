import runpy

import matplotlib

# The notebook sets pcv.params.debug = "plot", which makes plantcv/plantcv-geospatial show
# debug images via matplotlib. Inside a real Jupyter kernel that's harmless (an inline
# backend renders it into the notebook), but the converted script below runs as a plain
# process, where matplotlib would otherwise fall back to an interactive GUI backend (e.g.
# "macosx") and pop up a real window for every debug image, blocking the test. Force a
# non-interactive backend before anything else can import pyplot.
matplotlib.use("Agg", force=True)

import nbformat
import numpy as np
from nbconvert import PythonExporter


class _FakeShapesLayer:
    """Stand in for a Napari Shapes layer.

    The notebook is unmodified for testing, so there's no human available to draw a
    polygon when it's run this way. If nothing has populated `.data` by the time it's
    read, a placeholder polygon is used so downstream code has something valid to work
    with. Layers populated by the package itself before being read are unaffected.
    """

    _PLACEHOLDER = [[0, 0], [0, 100], [100, 100], [100, 0]]

    def __init__(self):
        self._data = []
        self.shape_type = []

    def _ensure_data(self):
        if not self._data:
            self._data = [np.array(self._PLACEHOLDER)]
            self.shape_type = ["polygon"]

    @property
    def data(self):
        self._ensure_data()
        return self._data

    @data.setter
    def data(self, value):
        self._data = [np.asarray(v) for v in value]
        self.shape_type = ["polygon"] * len(self._data)

    def add(self, data, shape_type="rectangle", **kwargs):
        for d in data:
            self._data.append(np.asarray(d))
            self.shape_type.append(shape_type)


class _FakePointsLayer:
    """Stand in for a Napari Points layer when no human is available to click points."""

    _PLACEHOLDER = [[0, 0], [0, 50], [50, 50], [50, 0], [25, 25], [10, 40]]

    def __init__(self):
        self._data = np.empty((0, 2))

    @property
    def data(self):
        if len(self._data) == 0:
            self._data = np.array(self._PLACEHOLDER, dtype=float)
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.asarray(value)


class _FakeViewer:
    """Minimal Napari viewer mock that prevents opening a GUI during the test."""

    def __init__(self, *args, **kwargs):
        self.layers = {}

    def add_image(self, *args, **kwargs):
        return None

    def add_shapes(self, name="Shapes", **kwargs):
        layer = _FakeShapesLayer()
        self.layers[name] = layer
        return layer

    def add_points(self, name="Points", **kwargs):
        layer = _FakePointsLayer()
        self.layers[name] = layer
        return layer

    def close(self):
        pass


class _FakeIPython:
    """Stand in for get_ipython() so notebook magics (e.g. %matplotlib widget) become no-ops."""

    def run_line_magic(self, *args, **kwargs):
        return None

    def run_cell_magic(self, *args, **kwargs):
        return None

    def system(self, *args, **kwargs):
        return None


def test_notebook(tmpdir, monkeypatch):
    """Convert and run the notebook as Python with Napari mocked to avoid a GUI."""
    import napari

    monkeypatch.setattr(napari, "Viewer", _FakeViewer)

    with open("index.ipynb", "r") as f:
        nb = nbformat.read(f, as_version=4)
    source, _ = PythonExporter().from_notebook_node(nb)

    script_path = tmpdir.join("index_converted.py")
    script_path.write(source)

    # Notebook-relative paths resolve from the repository root during pytest.
    runpy.run_path(
        str(script_path),
        init_globals={"get_ipython": _FakeIPython},
        run_name="__main__",
    )
