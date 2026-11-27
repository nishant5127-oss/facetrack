"""Public portal:  streamlit run mobile_app.py --server.port 8502"""
import os, tempfile, uuid
from html import escape
import cv2, streamlit as st
import db, face, matcher, emailer, ui

st.set_page_config(page_title="Missing Persons · Report a sighting", layout="centered")
ui.inject(); db.init(); os.makedirs("uploads", exist_ok=True)

# Tuning controls only appear when started with DEV_MODE=1
threshold = min(max(float(os.getenv("MATCH_THRESHOLD", 0.08)), 0.02), 0.30)
debug = False
if os.getenv("DEV_MODE") == "1":
    threshold = st.sidebar.slider("Match threshold (lower = stricter)", 0.02, 0.30, threshold, 0.005)
    debug = st.sidebar.checkbox("Show match distances (debug)")

CITIES = {"Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Bengaluru": (12.97, 77.59),
          "Kolkata": (22.57, 88.36), "Chennai": (13.08, 80.27), "Hyderabad": (17.39, 78.49),
          "Jaipur": (26.91, 75.79), "Ranchi": (23.34, 85.31), "Other": (None, None)}

ui.hero("Community alert", "Help bring someone home",
        "Browse people reported missing, or upload a photo if you think you have seen someone.")
board, report = st.tabs(["Missing persons", "Report a sighting"])

with board:
    cases = db.published_cases()
    if not cases:
        st.info("No public cases at the moment.")
    for c in cases:
        with ui.card():
            a, b = st.columns([1, 5])
            if c.photo_path and os.path.exists(c.photo_path):
                a.image(c.photo_path, width=150)
            b.markdown(f"<div class='name'>{escape(c.name)}</div>"
                       f"<div class='meta'>{c.age or '?'} yrs · last seen {escape(c.city or 'unknown')}</div>"
                       f"<p>{escape(c.description)}</p>", unsafe_allow_html=True)
    st.caption("If you see someone, use the Report a sighting tab. Please do not approach or confront anyone.")

with report:
    city = st.selectbox("Where was this taken?", list(CITIES))
    email = st.text_input("Your email (optional)")
    mode = st.radio("Input", ["Camera", "Photo", "Video"], horizontal=True)
    data, kind = None, None
    if mode == "Camera":
        f = st.camera_input("Take a photo"); data, kind = (f.getvalue() if f else None), "img"
    elif mode == "Photo":
        f = st.file_uploader("Photo", type=["jpg", "jpeg", "png"]); data, kind = (f.getvalue() if f else None), "img"
    else:
        f = st.file_uploader("Video", type=["mp4", "mov", "avi"]); data, kind = (f.getvalue() if f else None), "vid"

    if data and st.button("Submit report"):
        with st.spinner("Analysing…"):
            embs, imgs = [], []
            if kind == "img":
                e = face.embed_bytes(data)
                if e is not None:
                    embs, imgs = [e], [data]
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
                    t.write(data); p = t.name
                for fr in face.sample_frames(p):
                    e = face.embed(fr)
                    if e is not None:
                        embs.append(e); imgs.append(cv2.imencode(".jpg", fr)[1].tobytes())
                os.unlink(p)
            if not embs:
                st.error("No face detected. Try a clearer image."); st.stop()
            cases, best, best_img = db.active_cases(), None, None
            for e, im in zip(embs, imgs):
                if debug:
                    near = matcher.find_matches(e, cases, threshold=9.9, k=1)
                    if near:
                        st.caption(f"Debug: nearest = {near[0][0].name}, distance = {near[0][1]:.4f}, threshold = {threshold:.3f}")
                for m in matcher.find_matches(e, cases, threshold):
                    if best is None or m[2] > best[2]:
                        best, best_img = m, im
        if not best:
            st.info("No match found. Nothing from this upload has been stored.")
        else:
            case = best[0]
            saved = f"uploads/{uuid.uuid4().hex}.jpg"
            open(saved, "wb").write(best_img)
            lat, lon = CITIES[city]
            db.add(db.Sighting(case_id=case.id, city=city, lat=lat, lon=lon,
                               reporter_email=email, photo_path=saved, confidence=best[2]))
            sent = emailer.send(case.contact_email, f"Possible sighting: {case.name}",
                                f"A possible match for {case.name} was reported in {city} "
                                f"(score {best[2]}%). The sighting photo is attached. "
                                f"Please review it in the officer portal.", saved)
            if debug:
                st.caption(f"Debug: email sent = {sent}")
            st.success("Thank you. This may match a registered case and has been sent to officers for review.")
