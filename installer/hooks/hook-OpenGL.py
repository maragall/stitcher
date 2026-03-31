"""
Hook for PyOpenGL 3.x — from tlambert03/napari-pyinstaller reference.
Handles platform detection and OpenGL.arrays modules.
"""

from PyInstaller.compat import is_win, is_darwin
from PyInstaller.utils.hooks import collect_data_files, exec_statement
import os
import glob


def opengl_arrays_modules():
    """Return list of array modules for OpenGL module."""
    statement = 'import OpenGL; print(OpenGL.__path__[0])'
    opengl_mod_path = exec_statement(statement)
    arrays_mod_path = os.path.join(opengl_mod_path, 'arrays')
    files = glob.glob(arrays_mod_path + '/*.py')
    modules = []
    for f in files:
        mod = os.path.splitext(os.path.basename(f))[0]
        if mod == '__init__':
            continue
        modules.append('OpenGL.arrays.' + mod)
    return modules


if is_win:
    hiddenimports = ['OpenGL.platform.win32']
elif is_darwin:
    hiddenimports = ['OpenGL.platform.darwin']
else:
    hiddenimports = ['OpenGL.platform.glx']

hiddenimports += opengl_arrays_modules()

if is_win:
    datas = collect_data_files('OpenGL')
