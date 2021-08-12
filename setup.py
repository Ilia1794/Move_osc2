from distutils.core import Extension, setup
from Cython.Build import cythonize
import subprocess
import os
import numpy as np

# Start:
# python setup.py build_ext --inplace
ext = Extension(name="cython_module", sources = ["cython_module.pyx"])
setup(ext_modules=cythonize(ext))
subprocess.call(["python", "main.py"])
"""print("Очистить консоль?")
a = str(input())
if (a != 'Нет') and (a != 'Н') and (a != 'No') and (a != 'N') and (a != 'нет') and (a != 'н') and \
        (a != 'no') and (a != 'n') and (a != ' '):
    os.system(['cls'][os.name == os.sys.platform])
else:
    print('ok')"""
