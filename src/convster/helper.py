"""
To be added
"""
import os

def outfile_suffix(filename, suffix, separator:str='_'):
    # is_needed (only in `io::clip_to_ecoregion` which might not be needed
    """Insert suffix into filename and hand back basename_suffix.extension"""
    base, ext = os.path.splitext(filename)
    return f"{base}{separator}{suffix}{ext}"

