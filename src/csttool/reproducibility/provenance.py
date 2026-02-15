"""
Provenance tracking: Capture git hash, Python version, dependency versions.

Handles missing git gracefully (e.g., in installed packages or shallow clones).
"""

import os
import platform
import subprocess
import sys


def get_git_commit_hash() -> str | None:
    """Get git commit hash, handling missing git gracefully.

    Tries multiple approaches:
    1. Run `git rev-parse HEAD`
    2. Check environment variable GITHUB_SHA (CI environments)
    3. Return None if git unavailable or not in git repo

    Returns:
        Git commit hash as string, or None if unavailable
    """
    # Try git command
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback to environment variable (CI)
    if "GITHUB_SHA" in os.environ:
        return os.environ["GITHUB_SHA"]

    # Not available
    return None


def get_python_version() -> str:
    """Get Python version string."""
    return sys.version


def get_dependency_versions() -> dict:
    """Get versions of key dependencies.

    Returns:
        Dict mapping package name to version string
    """
    versions = {}

    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        versions["numpy"] = "not installed"

    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        versions["scipy"] = "not installed"

    try:
        import dipy
        versions["dipy"] = dipy.__version__
    except ImportError:
        versions["dipy"] = "not installed"

    try:
        import nibabel
        versions["nibabel"] = nibabel.__version__
    except ImportError:
        versions["nibabel"] = "not installed"

    return versions


def get_platform_info() -> dict:
    """Get platform information.

    Returns:
        Dict with platform, machine, processor info
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def get_provenance_dict() -> dict:
    """Get complete provenance information.

    Returns:
        Dict containing:
        - git_commit: Git commit hash or None
        - python_version: Python version string
        - dependencies: Dict of package versions
        - platform: Platform string
        - machine: Machine type
        - processor: Processor type
    """
    platform_info = get_platform_info()

    return {
        "git_commit": get_git_commit_hash(),
        "python_version": get_python_version(),
        "dependencies": get_dependency_versions(),
        "platform": platform_info["platform"],
        "machine": platform_info["machine"],
        "processor": platform_info["processor"],
    }
