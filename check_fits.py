from astropy.io import fits

file = r"PASTE_THE_PROBLEM_FILE_PATH_HERE"

hdul = fits.open(file)

print(hdul[1].columns.names)