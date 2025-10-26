from __future__ import annotations
import argparse
from . import __version__

def main() -> None:
    
    # Main parser
    parser = argparse.ArgumentParser(
        prog="csttool",
        description="CST assessment tool using DTI data."
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"        
        )
    
    # Subparser container
    # Defines an environment to add "subtools" to 
    subparsers = parser.add_subparsers(dest="command")
    
    # Dummy subtool to test functionality
    p_check = subparsers.add_parser("check", help="Run environment checks")
    p_check.set_defaults(func=cmd_check)
    
    args = parser.parse_args()    
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        
def cmd_check(args: argparse.Namespace) -> None:
    """Lightweight health check to confirm the CLI is wired correctly."""
    import sys
    print("csttool environment OK")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Version: {__version__}")
        
if __name__ == "__main__":
    main()