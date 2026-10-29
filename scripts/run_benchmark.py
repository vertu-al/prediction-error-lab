"""Thin script wrapper around the package CLI.

Use this after ``pip install -e '.[dev]'`` from the repository root.
"""

from prediction_error_lab.cli import main


if __name__ == "__main__":
    main()
