from templateflow import api as tflow

print("=" * 60)
print("TEMPLATEFLOW ATLAS DIAGNOSTIC")
print("=" * 60)

# 1. List all available templates
print("\n1. Available templates:")
templates = tflow.templates()
for t in sorted(templates)[:20]:  # First 20
    print(f"    {t}")
print(f"    ... ({len(templates)} total)")

# 2. Check what's available for MNI152NLin6Asym
print("\n2. MNI152NLin6Asym atlases:")
try:
    mni6_atlases = tflow.get('MNI152NLin6Asym', atlas=None, raise_empty=False)
    print(f"    Found: {mni6_atlases}")
except Exception as e:
    print(f"    Error: {e}")

# 3. Check MNI152NLin2009cAsym (common alternative)
print("\n3. MNI152NLin2009cAsym atlases:")
try:
    mni2009_atlases = tflow.get('MNI152NLin2009cAsym', atlas=None, raise_empty=False)
    if isinstance(mni2009_atlases, list):
        print(f"    Found {len(mni2009_atlases)} files")
        # Show unique atlas names
        atlas_names = set()
        for p in mni2009_atlases:
            parts = str(p).split('_')
            for part in parts:
                if part.startswith('atlas-'):
                    atlas_names.add(part)
        print(f"    Atlases: {atlas_names}")
except Exception as e:
    print(f"    Error: {e}")

# 4. Search specifically for Harvard-Oxford
print("\n4. Searching for Harvard-Oxford:")
for template in ['MNI152NLin6Asym', 'MNI152NLin2009cAsym', 'MNI152NLin6Sym']:
    try:
        result = tflow.get(template, atlas='HarvardOxford', raise_empty=False)
        if result:
            print(f"    {template}: Found!")
            if isinstance(result, list):
                for r in result[:5]:
                    print(f"        {r}")
        else:
            print(f"    {template}: Not found")
    except Exception as e:
        print(f"    {template}: Error - {e}")

# 5. List all atlases in MNI152NLin2009cAsym
print("\n5. All atlas files in MNI152NLin2009cAsym:")
try:
    all_files = tflow.get('MNI152NLin2009cAsym', raise_empty=False)
    if isinstance(all_files, list):
        # Extract unique atlas names
        atlases = {}
        for f in all_files:
            fname = str(f).split('/')[-1]
            if 'atlas-' in fname:
                for part in fname.split('_'):
                    if part.startswith('atlas-'):
                        atlas_name = part.replace('atlas-', '')
                        if atlas_name not in atlases:
                            atlases[atlas_name] = []
                        atlases[atlas_name].append(fname)
        
        print(f"    Found {len(atlases)} atlases:")
        for name, files in sorted(atlases.items()):
            print(f"      - {name}: {len(files)} files")
except Exception as e:
    print(f"    Error: {e}")

# 6. Check if any atlas has brainstem/subcortical labels
print("\n6. Looking for subcortical/brainstem atlases:")
for template in ['MNI152NLin2009cAsym', 'MNI152NLin6Asym']:
    try:
        result = tflow.get(template, desc='subcortical', raise_empty=False)
        if result:
            print(f"    {template} desc='subcortical': {result}")
    except:
        pass
    
    try:
        result = tflow.get(template, desc='sub', raise_empty=False)
        if result:
            print(f"    {template} desc='sub': {result}")
    except:
        pass