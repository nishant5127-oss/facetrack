"""Officer portal:  streamlit run Home.py"""
import json, os, secrets, time, uuid
from html import escape
import pandas as pd, streamlit as st, folium
from streamlit_folium import st_folium
import db, face, ui

st.set_page_config(page_title="Officer Portal · Missing Person Tracker", layout="wide")
ui.inject(); db.init(); os.makedirs("uploads", exist_ok=True)

@st.cache_resource
def _sessions():
    return {}  # token -> expiry time

def _token():
    return st.query_params.get("s", "")

def _logged_in():
    exp = _sessions().get(_token())
    return bool(exp and exp > time.time())

# ---------- login (stays signed in across page reloads) ----------
if not _logged_in():
    ui.hero("Officer portal", "Sign in", "Authorised personnel only.")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("login"):
            pw = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in")
        if ok:
            if pw == os.getenv("OFFICER_PASSWORD", "admin123"):
                tok = secrets.token_urlsafe(16)
                _sessions()[tok] = time.time() + 8 * 3600
                st.query_params["s"] = tok
                st.rerun()
            else:
                st.error("Wrong password")
    st.stop()

cases = db.all_cases()
sights = db.all_sightings()
by_id = {c.id: c for c in cases}
pending = [s for s in sights if s.case_id and s.review_status == "pending"]

with st.sidebar:
    st.markdown("### Missing Person Tracker")
    st.caption("Officer portal")
    st.markdown(f"**{sum(c.status == 'missing' for c in cases)}** active cases  \n"
                f"**{len(pending)}** matches to review  \n"
                f"**{sum(c.status == 'found' for c in cases)}** found")
    if st.button("Log out"):
        _sessions().pop(_token(), None)
        st.query_params.clear()
        st.rerun()

ui.hero("Officer portal", "Case management",
        "Register missing persons, review possible sightings and close cases.")
t1, t2, t3 = st.tabs(["Register case", "Cases", "Sightings"])

# ---------- register ----------
with t1:
    with st.form("case"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Full name")
        city = c2.text_input("Last seen city")
        c3, c4 = st.columns(2)
        age = c3.number_input("Age", 0, 120, 0)
        gender = c4.selectbox("Gender", ["", "Male", "Female", "Other"])
        email = st.text_input("Contact email (alerts go here)")
        desc = st.text_area("Description")
        photo = st.file_uploader("Clear front-facing photo", type=["jpg", "jpeg", "png"])
        publish = st.checkbox("Show on the public missing-persons board")
        go = st.form_submit_button("Register case")
    if go:
        if not (name and photo):
            st.error("Name and photo are required.")
        else:
            data = photo.getvalue()
            emb = face.embed_bytes(data)
            if emb is None:
                st.error("No face detected. Use a clearer front-facing photo.")
            else:
                path = f"uploads/{uuid.uuid4().hex}.jpg"
                open(path, "wb").write(data)
                db.add(db.Case(name=name, age=age or None, gender=gender, city=city,
                               contact_email=email, description=desc, photo_path=path,
                               embedding=json.dumps(emb.tolist()), published=publish))
                st.success(f"Case registered for {name}.")

# ---------- cases ----------
with t2:
    if not cases:
        st.info("No cases yet.")
    for c in cases:
        with ui.card():
            a, b = st.columns([1, 6])
            if c.photo_path and os.path.exists(c.photo_path):
                a.image(c.photo_path, width=140)
            kind = "red" if c.status == "missing" else "green"
            vis = ui.badge("PUBLIC", "amber") if c.published else ui.badge("PRIVATE", "gray")
            b.markdown(
                f"<span class='name'>{escape(c.name)}</span> &nbsp;"
                f"{ui.badge(c.status.upper(), kind)} {vis}"
                f"<div class='meta'>{c.age or '?'} · {c.gender or '?'} · last seen {escape(c.city or 'unknown')}</div>"
                f"<p>{escape(c.description)}</p>", unsafe_allow_html=True)
            x1, x2, x3, _ = st.columns([1.2, 1.2, 1, 3])
            if c.status == "missing" and x1.button("Mark found", key=f"f{c.id}"):
                db.mark_found(c.id); st.rerun()
            if x2.button("Unpublish" if c.published else "Publish", key=f"p{c.id}"):
                db.set_published(c.id, not c.published); st.rerun()
            if x3.button("Delete", key=f"d{c.id}"):
                st.session_state["confirm_del"] = c.id; st.rerun()
            if st.session_state.get("confirm_del") == c.id:
                st.warning(f"Permanently delete {c.name} and all linked sightings? This cannot be undone.")
                y, n, _ = st.columns([1.6, 1, 4])
                if y.button("Yes, delete permanently", key=f"y{c.id}"):
                    db.delete_case(c.id); st.session_state.pop("confirm_del"); st.rerun()
                if n.button("Cancel", key=f"n{c.id}"):
                    st.session_state.pop("confirm_del"); st.rerun()

# ---------- sightings ----------
with t3:
    m = folium.Map(location=[22.5, 79], zoom_start=5, tiles="OpenStreetMap")
    for s in sights:
        if s.case_id and s.review_status != "dismissed" and s.lat is not None:
            who = by_id[s.case_id].name if s.case_id in by_id else "Case"
            folium.Marker([s.lat, s.lon], popup=f"{who} · {s.city} · {s.confidence}%",
                          icon=folium.Icon(color="red" if s.review_status == "pending" else "green")).add_to(m)
    st_folium(m, height=420, width=None, returned_objects=[], key="sightings_map")

    st.subheader("Matches to review")
    if not pending:
        st.info("Nothing waiting for review.")
    for s in pending:
        c = by_id.get(s.case_id)
        if not c:
            continue
        with ui.card():
            st.markdown(
                f"**Possible match: {escape(c.name)}** &nbsp;{ui.badge(f'{s.confidence}% score', 'amber')}"
                f"<div class='meta'>{escape(s.city)} · {s.created_at:%d %b %Y, %H:%M} UTC · "
                f"reporter: {escape(s.reporter_email or 'not given')}</div>", unsafe_allow_html=True)
            a, b, d = st.columns([1, 1, 1.4])
            if c.photo_path and os.path.exists(c.photo_path):
                a.image(c.photo_path, caption="Registered", width=220)
            if s.photo_path and os.path.exists(s.photo_path):
                b.image(s.photo_path, caption="Sighting", width=220)
            choice = d.radio("Decision", ["Found", "Dismiss"], index=None, horizontal=True, key=f"r{s.id}")
            if d.button("Save decision", key=f"s{s.id}", disabled=choice is None):
                if choice == "Found":
                    db.set_review(s.id, "confirmed"); db.mark_found(c.id)
                else:
                    db.set_review(s.id, "dismissed")
                st.rerun()
    if sights:
        with st.expander("All sightings"):
            st.dataframe(pd.DataFrame([{
                "time": s.created_at, "city": s.city,
                "matched": by_id[s.case_id].name if s.case_id in by_id else "-",
                "score": s.confidence, "review": s.review_status} for s in sights]))
