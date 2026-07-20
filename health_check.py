from pathlib import Path


def main() -> None:
    app = Path("app.py")
    if not app.exists():
        raise SystemExit("app.py was not found in the repository root")
    compile(app.read_text(encoding="utf-8"), str(app), "exec")
    print("app.py syntax check passed")


if __name__ == "__main__":
    main()
