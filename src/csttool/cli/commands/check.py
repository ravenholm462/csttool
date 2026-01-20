
import argparse
import sys
from csttool import __version__

def cmd_check(args: argparse.Namespace) -> bool:
    """Runs environment checks. Returns True if all checks pass."""
    print("csttool environment check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Version: {__version__}")
    
    all_ok = True
    
    # Check key dependencies
    try:
        import dipy
        print(f"✓ DIPY: {dipy.__version__}")
    except ImportError:
        print("✗ DIPY: NOT FOUND")
        all_ok = False
    
    try:
        import nibabel
        print(f"✓ NiBabel: {nibabel.__version__}")
    except ImportError:
        print("✗ NiBabel: NOT FOUND")
        all_ok = False
    
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError:
        print("✗ NumPy: NOT FOUND")
        all_ok = False
    
    try:
        import scipy
        print(f"✓ SciPy: {scipy.__version__}")
    except ImportError:
        print("✗ SciPy: NOT FOUND")
        all_ok = False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib: NOT FOUND")
        all_ok = False
    
    try:
        import nilearn
        print(f"✓ Nilearn: {nilearn.__version__}")
    except ImportError:
        print("✗ Nilearn: NOT FOUND (needed for atlas)")
        all_ok = False
    
    # Check optional dependencies
    try:
        import weasyprint
        print(f"✓ Weasyprint: {weasyprint.__version__}")
    except ImportError:
        print("○ Weasyprint: NOT FOUND (optional, for PDF reports)")
    
    # Check ingest module
    try:
        from csttool.ingest import run_ingest_pipeline
        print("✓ Ingest module: available")
    except ImportError:
        print("○ Ingest module: not installed (legacy import will be used)")
    
    if all_ok:
        print("\n✓ All required dependencies available")
    else:
        print("\n✗ Some dependencies missing - install with: pip install -e .")
    
    return all_ok
