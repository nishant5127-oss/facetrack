import os, smtplib
from email.message import EmailMessage

def send(to: str, subject: str, body: str, attachment_path: str = None) -> bool:
    host, user, pw = os.getenv("SMTP_HOST"), os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not (host and user and pw and to):
        return False
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body)
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="sighting.jpg")
    try:
        with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", 587))) as s:
            s.starttls(); s.login(user, pw); s.send_message(msg)
        return True
    except Exception:
        return False
