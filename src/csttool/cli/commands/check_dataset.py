
import argparse
import json
from pathlib import Path

from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs

from csttool.ingest import assess_acquisition_quality
from csttool.tracking.modules.estimate_directions import get_max_sh_order

def cmd_check_dataset(args: argparse.Namespace) -> int:
    """Assess acquisition quality of a DWI dataset."""
    
    # 1. Locate files
    dwi_path = args.dwi
    if not dwi_path.exists():
        print(f"Error: DWI file not found: {dwi_path}")
        return 1
        
    # Try to find gradients if not provided
    bval_path = args.bval
    bvec_path = args.bvec
    
    if not bval_path or not bvec_path:
        try:
            if not bval_path:
                candidates = [
                    dwi_path.with_suffix('.bval'),
                    dwi_path.with_name(dwi_path.name.split('.')[0] + '.bval')
                ]
                if dwi_path.name.endswith('.nii.gz'):
                     candidates.append(dwi_path.with_name(dwi_path.name[:-7] + '.bval'))
                
                for c in candidates:
                    if c.exists():
                        bval_path = c
                        break
            
            if not bvec_path:
                 candidates = [
                    dwi_path.with_suffix('.bvec'),
                    dwi_path.with_name(dwi_path.name.split('.')[0] + '.bvec')
                ]
                 if dwi_path.name.endswith('.nii.gz'):
                     candidates.append(dwi_path.with_name(dwi_path.name[:-7] + '.bvec'))
                 
                 for c in candidates:
                    if c.exists():
                        bvec_path = c
                        break
                        
        except Exception:
            pass
            
    if not bval_path or not bval_path.exists():
        print("Error: Could not locate .bval file. Please provide --bval.")
        return 1
    if not bvec_path or not bvec_path.exists():
        print("Error: Could not locate .bvec file. Please provide --bvec.")
        return 1
        
    # 2. Load Data
    try:
        # Load NIfTI header for voxel size
        img = load_nifti(str(dwi_path), return_img=True)[2]
        header = img.header
        voxel_size = tuple(float(x) for x in header.get_zooms()[:3])
        
        # Load gradients
        bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    except Exception as e:
        print(f"Error loading files: {e}")
        return 1
        
    # Load JSON if available
    json_data = {}
    if args.json:
        if args.json.exists():
            try:
                with open(args.json, 'r') as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read JSON: {e}")
        else:
            print(f"Warning: JSON file not found: {args.json}")
            
    # 3. Assess Quality
    warnings_list = assess_acquisition_quality(
        bvecs=bvecs,
        bvals=bvals,
        voxel_size=voxel_size,
        bids_json=json_data
    )
    
    # 4. Generate Report
    print("=" * 80)
    print("                        CST TOOL - ACQUISITION QUALITY REPORT")
    print("=" * 80)
    
    print(f"\nSubject/File: {dwi_path.name}")
    print(f"Scan Date:    {json_data.get('AcquisitionDateTime', 'Unknown')}")
    
    print("\nACQUISITION PARAMETERS")
    print("-" * 22)
    
    # Recalculate basic stats for display
    b0_thr = 50
    dwi_mask = bvals > b0_thr
    n_directions = 0
    if dwi_mask.any():
        from numpy.linalg import norm
        import numpy as np
        vecs = bvecs[dwi_mask]
        # Normalize
        norms = norm(vecs, axis=1, keepdims=True)
        norms[norms==0] = 1
        vecs = vecs / norms
        n_directions = len(np.unique(np.round(vecs, decimals=4), axis=0))

    max_b = bvals.max()
    
    print(f"Gradient directions:     {n_directions}")
    print(f"Maximum b-value:         {max_b:.0f} s/mm²")
    print(f"Voxel size:              {voxel_size[0]:.2f} x {voxel_size[1]:.2f} x {voxel_size[2]:.2f} mm")
    if 'EchoTime' in json_data:
        try:
            et = float(json_data['EchoTime'])
            # Heuristic: if < 1 it's likely seconds, if > 1 it's likely ms (Dicom often ms)
            # BIDS spec says seconds.
            if et < 1.0:
                 et_ms = et * 1000
            else:
                 et_ms = et
            print(f"Echo time:               {et_ms:.1f} ms")
        except:
            pass
            
    if 'MultibandAccelerationFactor' in json_data:
        print(f"Multiband factor:        {json_data['MultibandAccelerationFactor']}")
        
    print("\nQUALITY ASSESSMENT")
    print("-" * 18)
    
    if not warnings_list:
        print("✅ No quality issues detected.")
    else:
        for severity, msg in warnings_list:
            if severity == "CRITICAL":
                icon = "❌"
            elif severity == "WARNING":
                icon = "⚠️ "
            else:
                icon = "ℹ️ "
            print(f"{icon} [{severity}] {msg}")
            
    print("\nRECOMMENDED SETTINGS")
    print("-" * 20)
    rec_sh = get_max_sh_order(n_directions)
    print(f"Maximum SH order:        {rec_sh}")
    
    suggested_step = min(voxel_size) * 0.5
    print(f"Suggested step size:     {suggested_step:.2f} mm")
    
    print("\n" + "=" * 80)
    
    return 0
