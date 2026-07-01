import os
import tempfile
import streamlit as st

from src.pipeline import analyze_star

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI-Enabled Exoplanet Detection",
    page_icon="🌌",
    layout="wide"
)

# ==========================================================
# STYLE
# ==========================================================

st.markdown("""
<style>

[data-testid="stMetric"]{
    background:#16213E;
    border:1px solid #2E4A7D;
    border-radius:15px;
    padding:15px;
}

[data-testid="stMetricValue"]{
    color:#4FC3F7;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.title("🌌 AI-Enabled Detection of Exoplanets")

st.markdown("""
Analyze NASA TESS light curves using

- ✅ Box Least Squares (BLS)
- ✅ Parameter Estimation
- ✅ AI Candidate Assessment
""")

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("Upload FITS Files")

    uploaded_files = st.file_uploader(
        "Choose one or more FITS files",
        type=["fits"],
        accept_multiple_files=True
    )

    analyze = st.button(
        "🚀 Analyze",
        use_container_width=True
    )

# ==========================================================
# WAIT
# ==========================================================

if not analyze:

    st.info("Upload one or more FITS files and click Analyze.")

    st.stop()

if not uploaded_files:

    st.warning("Please upload at least one FITS file.")

    st.stop()

# ==========================================================
# ANALYSIS
# ==========================================================

results = []

with st.spinner("Analyzing uploaded files..."):

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".fits"
        ) as tmp:

            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        star = analyze_star(temp_path)

        os.remove(temp_path)

        results.append(star)

results.sort(
    key=lambda x: x["params"]["power"],
    reverse=True
)

best_star = results[0]

params = best_star["params"]

# ==========================================================
# SUCCESS
# ==========================================================

st.success(f"Analyzed {len(results)} FITS file(s).")

st.success("🟢 Best Candidate Selected")

st.info(f"🏆 {params['file']}")

# ==========================================================
# PARAMETERS
# ==========================================================

st.subheader("🛰 Scientific Parameters")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Period", f"{params['period']:.4f} days")
    st.metric("Depth", f"{params['depth']:.6f}")

with c2:
    st.metric("Duration", f"{params['duration']:.4f} days")
    st.metric("SNR", f"{params['snr']:.2f}")

with c3:
    st.metric("Radius", f"{params['radius_rearth']:.2f} R⊕")
    st.metric("BLS Power", f"{params['power']:.2e}")

# ==========================================================
# PLOTS
# ==========================================================

st.divider()

st.subheader("📊 Detection Visualizations")

left, right = st.columns(2)

with left:

    if os.path.exists("outputs/plots/top_candidate.png"):

        st.image(
            "outputs/plots/top_candidate.png",
            caption="Raw Light Curve",
            use_container_width=True
        )

with right:

    if os.path.exists("outputs/plots/phase_folded.png"):

        st.image(
            "outputs/plots/phase_folded.png",
            caption="Phase Folded Light Curve",
            use_container_width=True
        )

# ==========================================================
# AI
# ==========================================================

st.divider()

st.subheader("🤖 AI Candidate Assessment")

a1, a2 = st.columns(2)

with a1:
    st.metric("Prediction", params["status"])

with a2:
    st.metric("Confidence", f"{params['confidence']:.2f}%")

st.caption(
    "Prototype Random Forest model trained on NASA Kepler data."
)

# ==========================================================
# CANDIDATE RANKING
# ==========================================================

st.divider()

st.subheader("🏆 Candidate Ranking")

ranking = []

for i, star in enumerate(results, start=1):

    p = star["params"]

    ranking.append({
        "Rank": i,
        "File": p["file"],
        "Period (days)": round(p["period"], 4),
        "Power": f"{p['power']:.2e}"
    })

st.dataframe(
    ranking,
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# RAW PARAMETERS
# ==========================================================

with st.expander("Show Raw Parameters"):

    st.json(params)