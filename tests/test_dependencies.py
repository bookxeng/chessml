import subprocess
import sys


def test_chessml_does_not_import_torch():
    code = "import sys, chessml; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
