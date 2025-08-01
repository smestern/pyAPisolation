# -*- mode: python ; coding: utf-8 -*-
import sys
import glob
import os

sys.setrecursionlimit(5000)
block_cipher = None
pf_foldr='C:\\Users\\SMest\\.conda\\envs\\pyinstaller\\Lib\\site-packages\\PySide2\\plugins\\platforms\\'

shiboken2_files = Tree("C:/Users/SMest/.conda/envs/pyinstaller/Lib\\site-packages\\shiboken2\\", prefix="shiboken2", excludes=['.pyc', 'tmp'])#glob.glob('C:/Users/SMest/.conda/envs/pyinstaller/Lib\\site-packages\\shiboken2\\**\\*', recursive=True)
##shiboken2_files = [(file, '.\\shiboken2'+''.join(file.split("shiboken2")[1:])) for file in shiboken2_files if ".dll" not in file]
#print(shiboken2_files)


hdmf_files  = Tree("C:\\Users\\SMest\\.conda\\envs\\pyinstaller\\Lib\\site-packages\\hdmf\\", prefix="hdmf", excludes=['.pyc', 'tmp'])
#hdmf_files = glob.glob("C:\\Users\\SMest\\.conda\\envs\\pyinstaller\\Lib\\site-packages\\hdmf\\**\\*",  recursive=True)
#hdmf_files = [(file, '.\\hdmf'+''.join(file.split("hdmf")[1:])) for file in hdmf_files if ".py" not in file]

pynwb_files = Tree("C:\\Users\\SMest\\.conda\\envs\\pyinstaller\\Lib\\site-packages\\pynwb\\", prefix="pynwb", excludes=['.pyc', 'tmp'])
#pynwb_files = glob.glob("C:\\Users\\SMest\\.conda\\envs\\pyinstaller\\Lib\\site-packages\\pynwb\\**\\*",  recursive=True)
#pynwb_files = [(file, '.\\pynwb'+''.join(file.split("pynwb")[1:])) for file in pynwb_files]

ipfx_files = "C:\\Users\\SMest\\.conda\\envs\\pyinstaller\\Lib\\site-packages\\ipfx\\version.txt"
ipfx_files = (ipfx_files, ".\\ipfx\\")

print(f" found {len(pynwb_files)}")

Prism_template = ("..\\dev\\prism_template2.pzfx", ".//pyAPisolation\\dev\\")
ui_file = ("..\\gui\\mainwindowMDI.ui", ".\\pyAPisolation\\gui\\")


a = Analysis(['run_spike_finder.py'],
             pathex=['C:\\Users\\SMest\\source\\repos\\smestern\\pyAPisolation\\pyAPisolation\\dev', "C:/Users/SMest/.conda/envs/pyinstaller/Lib\\site-packages\\shiboken2\\",
              "C:/Users/SMest/.conda/envs/pyinstaller/Lib\\site-packages\\PySide2\\", 
              "C:/Users/SMest/.conda/envs/pyinstaller/Library\\plugins\\platforms"],
             binaries=[(pf_foldr+'qwindows.dll', '.\\platforms\\qwindows.dll'),
             (pf_foldr+'qdirect2d.dll', '.\\platforms\\qdirect.dll'),
             (pf_foldr+'qoffscreen.dll', '.\\platforms\\qoffscreen.dll'),
             (pf_foldr+'qwebgl.dll', '.\\platforms\\qwebgl.dll')],
             datas=[('./run_rmp.spec', '.'), ipfx_files, Prism_template, ui_file],
             hiddenimports=['openpyxl.cell._writer', 'pkg_resources.extern', 'hdmf'],
             hookspath=[],
             runtime_hooks=['runqthook.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='main',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               shiboken2_files,
               hdmf_files,
               pynwb_files,
               strip=False,
               upx=True,
               upx_exclude=['qwindows.dll'],
               name='run_spike_finder')
