import os

from src.pipeline import analyze_star
from src.visualize import plot_top_candidate, plot_phase_folded
from src.export import export_results

folder = r"C:\AI Exoplanet Detection\data\raw"

fits_files = [
    os.path.join(folder, file)
    for file in os.listdir(folder)
    if file.endswith(".fits")
]

print(f"Found {len(fits_files)} FITS files\n")

results = []

for file in fits_files:
    star = analyze_star(file)
    results.append(star["params"])

results.sort(key=lambda x: x["power"], reverse=True)

export_results(results)

print("\nTop Candidate:", results[0]["file"], "\n")

for r in results:

    print(r["file"])

    print(f"  Period      : {r['period']:.4f} days")
    print(f"  Depth       : {r['depth']:.6f}")
    print(f"  Duration    : {r['duration']:.4f} days")
    print(f"  Power       : {r['power']:.6e}")
    print(f"  Radius      : {r['radius_rearth']:.2f} Earth radii")
    print(f"  Orbit       : {r['semi_major_axis_au']:.4f} AU")
    print(f"  AI Status   : {r['status']}")
    print(f"  Confidence  : {r['confidence']:.2f}%\n")

top_file = os.path.join(folder, results[0]["file"])
star = analyze_star(top_file)

plot_top_candidate(
    star["time"],
    star["flux"]
)

plot_phase_folded(
    star["time"],
    star["flux"],
    star["params"]["period"]
)