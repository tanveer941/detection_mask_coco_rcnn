# -*- mode: python -*-

block_cipher = None


a = Analysis(['rcnn_dt_ne.py'],
             pathex=['D:\\Work\\2018\\code\\Github_repo\\Detection_ecal\\mask_rcnn_service'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='rcnn_dt_ne',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
