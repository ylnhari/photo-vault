import sys
from pathlib import Path

# Add src/ to path so modules can import each other (e.g. indexer imports vision)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
