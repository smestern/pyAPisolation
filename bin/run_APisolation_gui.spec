# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)
block_cipher = None
pf_foldr='C:\\Users\\SMest\\Anaconda3\\Library\\plugins\\platforms\\'


a = Analysis(['run_APisolation_gui.py'],
             pathex=['C:\\Users\\SMest\\source\\repos\\smestern\\pyAPisolation\\pyAPisolation\\dev', "C:\\Users\\SMest\Anaconda3\\Lib\\site-packages\\shiboken2\\",
              "C:\\Users\\SMest\Anaconda3\\Lib\\site-packages\\PySide2\\", "C:\\Users\\SMest\Anaconda3\\Lib\\site-packages\\PyQt5\Qt\\", 
              "C:\\Users\\SMest\\Anaconda3\\Library\\plugins\\platforms"],
             binaries=[(pf_foldr+'qwindows.dll', 'qwindows.dll'),
             (pf_foldr+'qdirect2d.dll', 'qdirect.dll'),
             (pf_foldr+'qoffscreen.dll', 'qoffscreen.dll'),
             (pf_foldr+'qwebgl.dll', 'qwebgl.dll')],
             datas=[],
             hiddenimports=[],
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
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
