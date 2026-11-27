from contextlib import contextmanager
import streamlit as st

_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp li {font-family:'IBM Plex Sans',sans-serif;}
h1, h2, h3 {font-family:'Fraunces',serif !important;letter-spacing:-0.01em;color:#14213D;}
.block-container {padding-top:2rem;max-width:1100px;}
[data-testid="stToolbar"], .stAppDeployButton, footer {display:none !important;}
.hero {background:#14213D;border-radius:18px;padding:2rem 2.2rem;margin-bottom:1.6rem;}
.hero .eyebrow {letter-spacing:.16em;text-transform:uppercase;font-size:.72rem;color:#E8A87C;font-weight:600;}
.hero .title {font-family:'Fraunces',serif;font-size:2.3rem;font-weight:700;color:#F4F1EA;line-height:1.15;margin:.35rem 0 .45rem;}
.hero .sub {color:#C3CCDD;font-size:1rem;margin:0;}
.badge {display:inline-block;padding:.12rem .6rem;border-radius:999px;font-size:.7rem;font-weight:600;letter-spacing:.07em;vertical-align:middle;}
.badge.red {background:#FBE2DC;color:#9E2A18;}
.badge.green {background:#DCEFE2;color:#1F6B3A;}
.badge.gray {background:#E7E2D8;color:#555;}
.badge.amber {background:#FCEBC8;color:#855500;}
.name {font-family:'Fraunces',serif;font-size:1.45rem;font-weight:700;color:#14213D;line-height:1.25;display:inline;}
.meta {color:#6B6B6B;font-size:.9rem;margin:.15rem 0 .5rem;}
.card-marker {display:none;}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-marker) {background:#fff;border-radius:14px;padding:.9rem 1.1rem;}
.stButton>button, .stFormSubmitButton>button {background:#14213D;color:#F4F1EA;border:1px solid #14213D;border-radius:8px;font-weight:600;padding:.45rem 1.1rem;}
.stButton>button p, .stFormSubmitButton>button p {color:inherit !important;}
.stButton>button:hover, .stFormSubmitButton>button:hover {background:#C8412B;border-color:#C8412B;color:#fff;}
.stButton>button:disabled {background:#B9B4A8;border-color:#B9B4A8;color:#fff;}
[data-testid="stForm"] {background:#fff;border:1px solid #E1DACB;border-radius:14px;padding:1.4rem;}
.stTabs [data-baseweb="tab-list"] {gap:1.6rem;border-bottom:1px solid #D9D2C3;}
.stTabs [data-baseweb="tab"] {font-weight:600;}
[data-baseweb="tab-highlight"] {background:#C8412B !important;height:3px;}
section[data-testid="stSidebar"] {background:#14213D;}
section[data-testid="stSidebar"] * {color:#F4F1EA !important;}
section[data-testid="stSidebar"] .stButton>button {background:transparent;border:1px solid #F4F1EA;}
section[data-testid="stSidebar"] .stButton>button:hover {background:#C8412B;border-color:#C8412B;}
</style>"""

def inject():
    st.markdown(_CSS, unsafe_allow_html=True)

def hero(eyebrow, title, sub=""):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
                f'<div class="title">{title}</div><p class="sub">{sub}</p></div>',
                unsafe_allow_html=True)

def badge(text, kind="red"):
    return f'<span class="badge {kind}">{text}</span>'

@contextmanager
def card():
    with st.container(border=True):
        st.markdown('<span class="card-marker"></span>', unsafe_allow_html=True)
        yield

# ============================================================================
# GEOLOCATION & CITY MAPPING
# ============================================================================
CITIES = {
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
    "Bengaluru": (12.97, 77.59),
    "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27),
    "Hyderabad": (17.39, 78.49),
    "Jaipur": (26.91, 75.79),
    "Ranchi": (23.34, 85.31),
    "Other": (None, None)
}

def get_coordinates(city: str):
    """
    Returns (latitude, longitude) for a known city name, or (None, None) fallback for unknown cities.
    """
    if not isinstance(city, str):
        return (None, None)
    return CITIES.get(city, (None, None))

