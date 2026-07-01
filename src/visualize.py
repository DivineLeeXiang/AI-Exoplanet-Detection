import os
import numpy as np
import matplotlib.pyplot as plt


PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.show()
    plt.close()


def plot_periodogram(result):

    plt.figure(figsize=(12, 5))

    plt.plot(result.period, result.power, linewidth=1.5)

    plt.grid(alpha=0.3)

    plt.xlabel("Period (days)")
    plt.ylabel("BLS Power")
    plt.title("Box Least Squares Periodogram")

    save_plot("periodogram.png")


def plot_lightcurve(time, flux, title):

    plt.figure(figsize=(13, 5))

    plt.scatter(
        time,
        flux,
        s=2,
        alpha=0.6
    )

    plt.grid(alpha=0.3)

    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Flux")
    plt.title(title)

    save_plot("lightcurve.png")


def plot_detected_dips(time, flux, dip_time, dip_flux):

    plt.figure(figsize=(13, 5))

    plt.scatter(
        time,
        flux,
        s=2,
        alpha=0.5,
        label="Light Curve"
    )

    plt.scatter(
        dip_time,
        dip_flux,
        color="red",
        s=20,
        label="Detected Transit"
    )

    plt.grid(alpha=0.3)

    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Flux")
    plt.title("Detected Transit Candidates")

    plt.legend()

    save_plot("detected_dips.png")


def plot_top_candidate(time, flux):

    plt.figure(figsize=(13, 5))

    plt.scatter(
        time,
        flux,
        s=2,
        alpha=0.6
    )

    plt.grid(alpha=0.3)

    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Flux")
    plt.title("Top Candidate Light Curve")

    save_plot("top_candidate.png")


def plot_phase_folded(time, flux, period):

    phase = (time % period) / period

    order = np.argsort(phase)

    phase = phase[order]
    flux = flux[order]

    plt.figure(figsize=(10, 5))

    plt.scatter(
        phase,
        flux,
        s=3,
        alpha=0.6
    )

    plt.grid(alpha=0.3)

    plt.xlabel("Orbital Phase")
    plt.ylabel("Normalized Flux")
    plt.title(f"Phase Folded Light Curve (P = {period:.4f} days)")

    save_plot("phase_folded.png")