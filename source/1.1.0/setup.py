"""Builds the optional Cython/OpenMP extension river._expand.

Set RIVER_NO_CYTHON=1 to skip it. Any compilation failure is reported as a
warning; RIVER then falls back to the NumPy implementation.
"""
import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except Exception as e:  # pragma: no cover
            self.warn(f"Cython extension not built ({e}); NumPy fallback will be used")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:  # pragma: no cover
            self.warn(f"Cython extension {ext.name} not built ({e})")


ext_modules = []
if os.environ.get("RIVER_NO_CYTHON") != "1":
    try:
        import numpy as np
        from Cython.Build import cythonize

        if sys.platform == "win32":
            cargs, largs = ["/O2", "/openmp"], []
        elif sys.platform == "darwin":
            # Apple clang: OpenMP needs libomp (conda: llvm-openmp)
            cargs, largs = ["-O3", "-Xpreprocessor", "-fopenmp"], ["-lomp"]
            if os.environ.get("RIVER_NO_OPENMP") == "1":
                cargs, largs = ["-O3"], []
        else:
            cargs, largs = ["-O3", "-fopenmp"], ["-fopenmp"]
        ext_modules = cythonize(
            [Extension("river._expand", ["src/river/_expand.pyx"],
                       include_dirs=[np.get_include()],
                       define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                       extra_compile_args=cargs, extra_link_args=largs)],
            language_level=3)
    except ImportError:
        print("Cython/numpy not available at build time; skipping river._expand")

setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
