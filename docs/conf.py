from importlib.metadata import version as _version

project = "pyfilter"
release = _version("pyfilter")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_autodoc_typehints",
]
# Do NOT also enable sphinx.ext.napoleon; it conflicts with numpydoc.

html_theme = "pydata_sphinx_theme"

# autosummary / numpydoc
autosummary_generate = True
numpydoc_show_class_members = False  # avoids duplicate-entry warnings with autosummary
numpydoc_class_members_toctree = False

# Docstring validation at build time (warnings; errors under -W)
numpydoc_validation_checks = {"all", "GL01", "SA01", "EX01", "ES01"}
# With "all" in the set, the other codes are *excluded*.
# Start permissive and tighten as the codebase conforms.

# Cross-references to NumPy/SciPy/Python docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

nitpicky = True  # warn on every unresolved reference
