from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# napari uses lazy_loader — PyInstaller can't trace lazy imports
hiddenimports = collect_submodules('napari')

# napari_builtins is a separate package with required data files (builtins.yaml)
hiddenimports += collect_submodules('napari_builtins')

# napari-svg plugin (installed as napari_svg)
try:
    hiddenimports += collect_submodules('napari_svg')
except Exception:
    pass

# Collect data files (icons, stylesheets, manifests, yaml configs)
datas = collect_data_files('napari')
datas += collect_data_files('napari_builtins')

try:
    datas += collect_data_files('napari_svg')
except Exception:
    pass
