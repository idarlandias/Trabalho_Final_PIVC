"""
Script para gerar executável Windows com PyInstaller.

Uso:
    pip install pyinstaller
    python build_exe.py
"""

import subprocess
import sys
import os

APP_NAME = "ClassificadorPneumonia"
MAIN_SCRIPT = "app.py"
MODEL_FILE = os.path.join("results", "exp3", "best_model.pth")
ICON = None  # Pode definir um .ico aqui se quiser

def main():
    if not os.path.exists(MODEL_FILE):
        print(f"ERRO: Modelo não encontrado: {MODEL_FILE}")
        print("Execute 'python train.py --exp exp3' primeiro.")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--add-data", f"{MODEL_FILE}{os.pathsep}results/exp3",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "torchvision.models",
        "--hidden-import", "PIL",
        "--collect-data", "torch",
        "--collect-data", "torchvision",
        MAIN_SCRIPT,
    ]

    if ICON and os.path.exists(ICON):
        cmd.extend(["--icon", ICON])

    print("=" * 60)
    print(f"Gerando executável: {APP_NAME}")
    print("=" * 60)
    print(f"Comando: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        dist_path = os.path.join("dist", APP_NAME)
        print("\n" + "=" * 60)
        print("EXECUTAVEL GERADO COM SUCESSO!")
        print(f"Local: {os.path.abspath(dist_path)}")
        print(f"Execute: {os.path.join(dist_path, APP_NAME + '.exe')}")
        print("=" * 60)
    else:
        print("\nERRO ao gerar executável. Verifique o log acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
