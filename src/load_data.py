from astropy.io import fits


def load_lightcurve(filepath):
    """
    Load a TESS/Kepler light curve from a FITS file.

    Supports:
    - PDCSAP_FLUX
    - SAP_FLUX
    """

    hdul = fits.open(filepath)

    if len(hdul) < 2:
        raise ValueError(
            "This FITS file does not contain a light curve table."
        )

    data = hdul[1].data

    columns = data.columns.names

    # -------------------------
    # Check TIME
    # -------------------------

    if "TIME" not in columns:
        raise ValueError(
            f"'TIME' column not found.\nAvailable columns:\n{columns}"
        )

    # -------------------------
    # Detect Flux Column
    # -------------------------

    if "PDCSAP_FLUX" in columns:

        flux_column = "PDCSAP_FLUX"

    elif "SAP_FLUX" in columns:

        flux_column = "SAP_FLUX"

    else:

        raise ValueError(
            "No supported flux column found.\n\n"
            f"Available columns:\n{columns}"
        )

    # -------------------------
    # Detect QUALITY
    # -------------------------

    if "QUALITY" in columns:

        quality = data["QUALITY"]

    else:

        print("Warning: QUALITY column not found. Using zeros.")

        quality = [0] * len(data)

    return {
        "TIME": data["TIME"],
        "FLUX": data[flux_column],
        "QUALITY": quality,
        "FLUX_COLUMN": flux_column
    }