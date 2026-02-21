"""Entry point for `python -m nexus.train`. Avoids sys.modules warning from -m nexus.train.main."""

import setuptools  # noqa: F401 - load setuptools before deps to avoid distutils deprecation warning

from .main import main

if __name__ == "__main__":
    main()
