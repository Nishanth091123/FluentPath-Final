import re
import base64
import json
from pathlib import Path
import streamlit as st
import difflib
import string
import threading
import io
import hashlib
import wave
import requests

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

from datetime import datetime, timezone, timedelta

from sqlalchemy import select, Column, Integer, String, Text, DateTime

from database.connection import Base, engine, SessionLocal
from database.models import User, ActivityLog

# Camera / microphone
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ai_engine import ask_ai, _setting
from techprep_rag import build_rag_user_prompt
from techprep_langchain import generate_techprep_answer
from techprep_langgraph import generate_techprep_graph_answer
from adaptive_history import build_user_adaptive_profile
from techprep_agent import run_techprep_agent
from interview_intelligence_v2 import run_interview_intelligence
from score_scale_guardrail import validate_score_contract
from learning_memory import build_learning_memory


IST = timezone(timedelta(hours=5, minutes=30))


class ActivityReview(Base):
    """Full review payload for new learning activities.

    Kept in a separate table so the existing ActivityLog schema and old data
    do not need to be changed.
    """
    __tablename__ = "activity_reviews"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    activity_log_id = Column(Integer, nullable=False, index=True, unique=True)
    user_id = Column(String(50), nullable=False, index=True)
    activity_type = Column(String(50), nullable=False, index=True)
    review_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


def save_activity_review(db, activity_log, review_data):
    if not review_data or activity_log is None:
        return
    payload = json.dumps(review_data, ensure_ascii=False, default=str)
    row = ActivityReview(
        activity_log_id=activity_log.id,
        user_id=activity_log.user_id,
        activity_type=activity_log.activity_type,
        review_json=payload
    )
    db.add(row)


def load_activity_review_map(user_id, log_ids):
    if not log_ids:
        return {}
    db = SessionLocal()
    try:
        rows = (
            db.query(ActivityReview)
            .filter(
                ActivityReview.user_id == user_id,
                ActivityReview.activity_log_id.in_(log_ids)
            )
            .all()
        )
        result = {}
        for row in rows:
            try:
                result[row.activity_log_id] = json.loads(row.review_json)
            except Exception:
                result[row.activity_log_id] = {"note": "Saved review data could not be decoded."}
        return result
    finally:
        db.close()



# ============================================================
# STEP 13 - FACE-TO-FACE CAMERA POSITION ANALYZER
# ============================================================

class FaceToFaceAnalyzer:
    """Measures visible face presence/position. It does NOT measure true eye contact."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()
        self.detector = None

        if cv2 is not None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.detector = cv2.CascadeClassifier(cascade_path)
                if self.detector.empty():
                    self.detector = None
            except Exception:
                self.detector = None

    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.frames = 0
            self.face_frames = 0
            self.centered_frames = 0
            self.good_distance_frames = 0
            self.too_far_frames = 0
            self.too_close_frames = 0
            self.off_center_frames = 0
            self.no_face_frames = 0
            self.diversion_events = 0
            self._was_diverted = False

    def process(self, frame):
        """Analyze live frames while keeping the interview video clean."""
        if cv2 is None or self.detector is None:
            return frame
        try:
            image = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
            )
            height, width = gray.shape[:2]
            with self.lock:
                self.frames += 1
                if len(faces) == 0:
                    self.no_face_frames += 1
                    diverted = True
                else:
                    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                    self.face_frames += 1
                    face_cx, face_cy = x + w / 2, y + h / 2
                    frame_cx, frame_cy = width / 2, height / 2
                    x_offset = abs(face_cx - frame_cx) / max(width, 1)
                    y_offset = abs(face_cy - frame_cy) / max(height, 1)
                    centered = x_offset <= 0.18 and y_offset <= 0.22
                    face_ratio = (w * h) / max(width * height, 1)
                    too_far = face_ratio < 0.055
                    too_close = face_ratio > 0.34
                    good_distance = not too_far and not too_close

                    if centered:
                        self.centered_frames += 1
                    else:
                        self.off_center_frames += 1
                    if too_far:
                        self.too_far_frames += 1
                    elif too_close:
                        self.too_close_frames += 1
                    else:
                        self.good_distance_frames += 1
                    diverted = not centered

                if diverted and not self._was_diverted:
                    self.diversion_events += 1
                self._was_diverted = diverted
        except Exception:
            pass
        return frame


    def summary(self):
        with self.lock:
            frames = self.frames
            if frames <= 0:
                return {
                    "available": False,
                    "presence": 0,
                    "centered": 0,
                    "distance": 0,
                    "camera_presence_score": 0,
                    "diversions": 0,
                    "too_far": 0,
                    "too_close": 0,
                    "off_center": 0,
                }

            presence = round(100 * self.face_frames / frames)
            centered = round(100 * self.centered_frames / max(self.face_frames, 1))
            distance = round(100 * self.good_distance_frames / max(self.face_frames, 1))

            # Camera Position/Presence score only; not an eye-contact score.
            camera_score = round(
                0.45 * presence +
                0.35 * centered +
                0.20 * distance
            )

            return {
                "available": True,
                "presence": presence,
                "centered": centered,
                "distance": distance,
                "camera_presence_score": max(0, min(100, camera_score)),
                "diversions": self.diversion_events,
                "too_far": self.too_far_frames,
                "too_close": self.too_close_frames,
                "off_center": self.off_center_frames,
            }


def face_to_face_feedback(stats):
    """Create factual feedback from measured camera-position statistics."""
    if not stats.get("available"):
        return (
            "Camera analysis data is not available yet. Keep the camera running "
            "during the interview to receive face-to-face feedback."
        )

    messages = []
    score = stats["camera_presence_score"]

    if score >= 85 and stats["diversions"] <= 3:
        messages.append(
            "You maintained a very stable face-to-face interview position with very few diversions."
        )
    elif score >= 70:
        messages.append(
            "You maintained a good camera position overall, with a few moments that can be improved."
        )
    else:
        messages.append(
            "Your camera presence needs more consistency. Try to remain visible and centered throughout the interview."
        )

    if stats["presence"] < 80:
        messages.append("Your face was not clearly visible for part of the interview.")
    elif stats["presence"] >= 95:
        messages.append("Your face remained visible for almost the complete interview.")

    if stats["centered"] < 70:
        messages.append("You moved away from the center frequently. Keep your face closer to the middle of the camera frame.")
    elif stats["centered"] >= 90:
        messages.append("Your face stayed well centered for most of the interview.")

    if stats["too_far"] > stats["too_close"] and stats["distance"] < 75:
        messages.append("You appeared far from the camera at times. Move slightly closer.")
    elif stats["too_close"] > stats["too_far"] and stats["distance"] < 75:
        messages.append("You appeared too close to the camera at times. Move slightly back.")
    elif stats["distance"] >= 85:
        messages.append("Your distance from the camera was generally good.")

    if stats["diversions"] >= 8:
        messages.append("There were several camera-position diversions. Try to reduce side movements and distractions.")
    elif stats["diversions"] <= 3 and score >= 75:
        messages.append("There were no major camera-position diversions.")

    return " ".join(messages)


# ============================================================
# PAGE CONFIG
# ============================================================

# ============================================================
# STEP 3E - VOICE ACCURACY HELPERS
# ============================================================

def _normalize_voice_words(value):
    """
    Normalize harmless speech-to-text formatting differences before comparison.
    Example: "every day" and "everyday" are treated as equivalent.
    """
    if not value:
        return []

    normalized = value.lower()
    table = str.maketrans("", "", string.punctuation)
    normalized = normalized.translate(table)

    # Common STT joined/split word variants.
    phrase_equivalents = {
        "every day": "everyday",
        "any one": "anyone",
        "some one": "someone",
        "no one": "noone",
        "some thing": "something",
        "any thing": "anything",
        "every thing": "everything",
    }

    for phrase, canonical in phrase_equivalents.items():
        normalized = re.sub(r"\b" + re.escape(phrase) + r"\b", canonical, normalized)

    return normalized.split()


def calculate_voice_accuracy(expected_text, heard_text):
    """Word/transcript accuracy only; NOT a phoneme pronunciation score."""
    expected_words = _normalize_voice_words(expected_text)
    heard_words = _normalize_voice_words(heard_text)

    if not expected_words:
        return {"score": 0, "correct": 0, "total": 0, "missing": [], "different": []}

    matcher = difflib.SequenceMatcher(None, expected_words, heard_words)
    correct, missing, different = 0, [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            correct += i2 - i1
        elif tag == "delete":
            missing.extend(expected_words[i1:i2])
        elif tag == "replace":
            different.append((" ".join(expected_words[i1:i2]), " ".join(heard_words[j1:j2])))
        elif tag == "insert":
            different.append(("(extra)", " ".join(heard_words[j1:j2])))

    score = round((correct / len(expected_words)) * 100)
    return {
        "score": max(0, min(100, score)),
        "correct": correct,
        "total": len(expected_words),
        "missing": missing,
        "different": different,
    }



# ============================================================
# STEP 4B - ADAPTIVE SPEAKING COACH HELPERS
# ============================================================

def get_adaptive_focus(ai_report):
    """Find the lowest score in the existing AI speaking report."""
    if not ai_report:
        return None, None

    score_labels = {
        "Grammar": ["grammar"],
        "Sentence Formation": ["sentence formation"],
        "Vocabulary": ["vocabulary"],
        "Clarity": ["clarity"],
        "Topic Relevance": ["topic relevance", "relevance"],
    }

    found_scores = {}

    for display_name, labels in score_labels.items():
        for label in labels:
            match = re.search(
                rf"{re.escape(label)}\s*[:\-]\s*(\d{{1,2}})\s*/\s*20",
                ai_report,
                flags=re.IGNORECASE,
            )
            if match:
                found_scores[display_name] = max(
                    0, min(20, int(match.group(1)))
                )
                break

    if not found_scores:
        return None, None

    focus_area = min(found_scores, key=found_scores.get)
    return focus_area, found_scores[focus_area]


def adaptive_coach_tip(focus_area):
    tips = {
        "Grammar": (
            "Use short, complete sentences and keep your tense consistent."
        ),
        "Sentence Formation": (
            "Use this structure in your next answer: main point → reason → example."
        ),
        "Vocabulary": (
            "Use 2–3 useful topic-related words naturally in your next answer."
        ),
        "Clarity": (
            "Pause between ideas and keep each sentence focused on one point."
        ),
        "Topic Relevance": (
            "Answer the question directly first, then add a reason and one example."
        ),
    }
    return tips.get(
        focus_area,
        "Focus on one clear improvement in your next speaking answer."
    )


class LiveInterviewAudioCollector:
    """Collect PCM from the same WebRTC camera/microphone session."""

    def __init__(self):
        self.lock = threading.Lock()
        self.recording = False
        self.resampler = None
        self.chunks = []
        self.error = None
        self.total_bytes = 0
        self.frames_seen = 0

    def start(self):
        from av import AudioResampler
        with self.lock:
            self.resampler = AudioResampler(format="s16", layout="mono", rate=16000)
            self.chunks = []
            self.total_bytes = 0
            self.frames_seen = 0
            self.error = None
            self.recording = True

    def _add_frames(self, frames):
        if frames is None:
            return
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        for converted in frames:
            chunk = converted.to_ndarray().tobytes()
            self.chunks.append(chunk)
            self.total_bytes += len(chunk)

    def process(self, frame):
        with self.lock:
            if not self.recording:
                return frame
            try:
                self.frames_seen += 1
                self._add_frames(self.resampler.resample(frame))
                if self.total_bytes > 16000 * 2 * 300:
                    self.error = "Recording exceeded five minutes."
                    self.recording = False
            except Exception as exc:
                self.error = f"Microphone capture failed ({type(exc).__name__})."
                self.recording = False
        return frame

    def stop_wav(self):
        with self.lock:
            self.recording = False
            if self.error:
                message = self.error
                self.clear_locked()
                raise RuntimeError(message)
            if self.resampler is not None:
                try:
                    self._add_frames(self.resampler.resample(None))
                except Exception:
                    pass
            pcm = b"".join(self.chunks)
            self.clear_locked()
        if len(pcm) < 16000:  # At least half a second at 16 kHz, 16-bit mono.
            raise ValueError("No clear microphone audio was captured. Check mic permission.")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm)
        return output.getvalue()

    def clear_locked(self):
        self.resampler = None
        self.chunks = []
        self.total_bytes = 0
        self.error = None

    def status(self):
        with self.lock:
            return self.frames_seen, self.total_bytes, self.error

    def clear(self):
        with self.lock:
            self.recording = False
            self.clear_locked()


def transcribe_interview_audio(audio_bytes):
    """Transcribe the complete answer; fall back to the existing recognizer."""
    groq_key = _setting("GROQ_API_KEY")
    groq_error = None
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {groq_key}"},
                files={"file": ("interview_answer.wav", audio_bytes, "audio/wav")},
                data={
                    "model": _setting("GROQ_STT_MODEL", "whisper-large-v3"),
                    "language": "en",
                    "response_format": "json",
                    "temperature": "0",
                    "prompt": "English job interview. Technical Services, civil engineering, cement industry.",
                },
                timeout=90,
            )
            response.raise_for_status()
            text = response.json().get("text", "").strip()
            if text:
                return text, "Groq Whisper"
            groq_error = "empty transcript"
        except (requests.RequestException, ValueError, TypeError) as exc:
            groq_error = type(exc).__name__

    if sr is None:
        raise RuntimeError(
            "SpeechRecognition is unavailable" +
            (f"; Groq transcription: {groq_error}" if groq_error else ".")
        )
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        speech = recognizer.record(source)
    text = recognizer.recognize_google(speech, language="en-IN").strip()
    return text, "Google speech recognition"


st.set_page_config(
    page_title="FluentPath | English & Interview Coach",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# STEP 14 - FRIENDLY COACH EXPERIENCE
# ============================================================

FRIENDLY_PAGE_MESSAGES = {
    "🏠 Dashboard": "🌟 Welcome back! Every small practice is building your confidence.",
    "📚 Learning": "🌱 Learn at your own pace. Small steps become strong skills.",
    "📘 Grammar": "💛 Grammar becomes easier with practice. Mistakes are part of learning.",
    "⏱️ Tenses": "🌿 Take one tense at a time. You do not need to master everything today.",
    "🧠 Vocabulary": "✨ Every new word gives you one more way to express yourself.",
    "✍️ Sentence Formation": "🧩 Start simple, then build your idea step by step.",
    "📝 Practice": "🤝 This is a safe place to try, learn and try again.",
    "🧪 Tests": "🎯 This is practice, not pressure. Use each question to learn something useful.",
    "📝 Daily Assignment": "🌟 Give it a try in your own words. Your coach is here to help you improve.",
    "🗣️ Speaking": "🎙️ Speak naturally. You do not need perfect English to make progress.",
    "🎤 Interviews": "🤝 Treat this like friendly practice. Pause, think and answer in your own way.",
    "🧑‍💻 TechPrep": "🚀 Understand the concept, structure your answer, practise it, and improve step by step.",
    "💬 SmartSpeak": "💬 Think naturally in your own way. SmartSpeak will help you express the same idea clearly in English.",
    "📈 Progress": "📈 Progress is more than one score. Consistency and retrying matter too.",
    "📜 History": "🌱 Every saved activity is part of your learning journey."
}

def friendly_coach_message(page_name):
    st.info(FRIENDLY_PAGE_MESSAGES.get(page_name, "🌟 Keep going — one small improvement at a time."))

def reset_counter(counter_key):
    st.session_state[counter_key] = st.session_state.get(counter_key, 0) + 1
    st.rerun()

def friendly_score_message(score):
    try:
        score = int(score)
    except Exception:
        return "Keep practicing — every attempt helps."
    if score >= 85:
        return "🌟 Great progress! Keep this confidence and consistency."
    if score >= 70:
        return "👏 Nice work! A little more practice can make this even stronger."
    if score >= 50:
        return "🌱 Good start! Focus on one improvement and try again."
    return "🤝 You have started — that matters. Keep it simple, use the coach tip, and try again."

def english_practice_language_check(text):
    """Friendly local guard for English-practice answers."""
    text = (text or "").strip()
    if not text:
        return True, ""

    tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", text))
    latin_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    alpha_chars = sum(ch.isalpha() for ch in text)

    tamil_answer = tamil_chars >= 3
    mostly_non_latin = alpha_chars >= 8 and len(latin_words) < 2

    if tamil_answer or mostly_non_latin:
        return False, (
            "🌱 Nice try! I understand your idea, but this activity is for English practice. "
            "No score will be given for this attempt. Try the same idea in simple English — "
            "even 2 or 3 short sentences are enough."
        )
    return True, ""

def show_english_retry_support(text):
    ok, message = english_practice_language_check(text)
    if not ok:
        st.info(message)
        st.markdown("**💡 Easy English structure:** Main point → Reason → Example → Result")
        st.caption("Think of the idea in Tamil if that helps, then express the same idea in simple English.")
    return ok

def voice_transcript_is_english(text):
    """English-only gate before voice text is displayed or scored."""
    text = (text or "").strip()
    if not text:
        return False
    local_ok, _ = english_practice_language_check(text)
    if not local_ok:
        return False
    try:
        verdict = ask_ai(
            "You are a strict language detector. Reply exactly ENGLISH if the text is primarily English. Otherwise reply exactly OTHER. Do not translate or explain.",
            text
        )
        return (verdict or "").strip().upper().startswith("ENGLISH")
    except Exception:
        return local_ok

def show_english_voice_message():
    st.info("🎙️ Please answer in English. Simple English is completely fine. Your non-English recording was not saved or scored.")



# ============================================================
# DATABASE
# ============================================================


from sqlalchemy import text

with engine.begin() as conn:
    if engine.dialect.name == "postgresql":
        conn.execute(text("SELECT pg_advisory_xact_lock(72638142)"))
    Base.metadata.create_all(bind=conn)
  



# ============================================================
# STEP 17C FINAL - LOGIN HERO ASSET
# ============================================================

def fluentpath_login_hero_data_uri():
    """Load the bundled FluentPath login visual without external URLs."""
    candidates = [
        Path(__file__).with_name("fluentpath_login_hero.png"),
        Path.cwd() / "fluentpath_login_hero.png",
    ]

    for hero_path in candidates:
        try:
            if hero_path.exists():
                encoded = base64.b64encode(hero_path.read_bytes()).decode("ascii")
                return f"data:image/png;base64,{encoded}"
        except Exception:
            pass

    return ""

# ============================================================
# SESSION
# ============================================================

def initialize_session():

    defaults = {
        "logged_in": False,
        "user_id": None,
        "user_name": None,
        "is_admin": False,
        "assignment_submitted": False,
        "test_score": 0,
        "practice_score": 0,
        "speaking_report": "",
        "step4_retry_mode": False,
        "step4_baseline_score": None,
        "step4_baseline_focus": None,
        "step4_baseline_question": None,
        "step4_retry_result_score": None,
        "step5_interview_scores": [],
        "step5_interview_reports": [],
        "step5_interview_questions": [],
        "interview_started": False,
        "interview_question": "",
        "interview_history": [],
        "interview_feedback": "",
        "interview_answer": "",
        "interview_count": 0,
        "final_interview_report": "",
        "interview_logged": False,
        "step13_interview_mode": "Text Practice",
        "step13_camera_analyzer": None,
        "step13_last_camera_summary": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


def login_user(user):

    st.session_state.logged_in = True
    st.session_state.user_id = user.user_id
    st.session_state.user_name = user.name
    st.session_state.is_admin = False


def logout_user():

    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_name = None
    st.session_state.is_admin = False


initialize_session()
# ============================================================
# ACTIVITY TRACKING
# ============================================================

def save_activity(activity_type, activity, score=None, review_data=None):

    if not st.session_state.get("user_id"):
        return False

    user_id = st.session_state.user_id
    db = SessionLocal()

    try:
        latest = db.execute(
            select(ActivityLog)
            .where(
                ActivityLog.user_id == user_id,
                ActivityLog.activity_type == activity_type,
                ActivityLog.activity == activity
            )
            .order_by(ActivityLog.created_at.desc())
        ).scalars().first()

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

        if latest and latest.created_at:
            age_seconds = (now_utc - latest.created_at).total_seconds()

            # One Daily Assignment record per IST calendar day.
            if activity_type == "Assignment":
                latest_ist_date = (
                    latest.created_at + timedelta(hours=5, minutes=30)
                ).date()
                current_ist_date = (
                    now_utc + timedelta(hours=5, minutes=30)
                ).date()

                if latest_ist_date == current_ist_date:
                    return False

            # Protect accidental Test double-click / rerun.
            elif activity_type == "Test":
                if latest.score == score and age_seconds < 60:
                    return False

            # Protect immediate duplicate Speaking analysis.
            elif activity_type == "Speaking":
                if age_seconds < 60:
                    return False

            # Protect repeated Interview final-report reruns.
            elif activity_type == "Interview":
                if age_seconds < 600:
                    return False

        log = ActivityLog(
            user_id=user_id,
            activity_type=activity_type,
            activity=activity,
            score=score
        )

        db.add(log)
        db.flush()  # get activity log ID before saving the optional review payload
        save_activity_review(db, log, review_data)
        db.commit()
        return True

    except Exception as e:
        db.rollback()
        print(f"Activity logging error: {e}")
        return False

    finally:
        db.close()

# ============================================================
# USER AUTHENTICATION
# ============================================================

COMMON_PASSWORD = "2026"
ADMIN_USER_ID = "NISHANTH"
ADMIN_PASSWORD = "091123"


def authenticate_admin(admin_id, password):
    return (
        admin_id.strip().upper() == ADMIN_USER_ID
        and password == ADMIN_PASSWORD
    )


def generate_user_id(db):

    last_user = db.execute(
        select(User)
        .order_by(User.id.desc())
    ).scalars().first()

    if last_user is None:
        return "NS001"

    next_number = last_user.id + 1

    return f"NS{next_number:03d}"


def create_user(name):

    name = name.strip()

    if not name:
        return None, "Please enter your name."

    db = SessionLocal()

    try:

        user_id = generate_user_id(db)

        new_user = User(
            user_id=user_id,
            name=name
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return new_user, None

    except Exception as e:

        db.rollback()

        return None, str(e)

    finally:

        db.close()


def authenticate_user(user_id, password):

    user_id = user_id.strip().upper()

    if password != COMMON_PASSWORD:
        return None

    db = SessionLocal()

    try:

        user = db.execute(
            select(User).where(
                User.user_id == user_id
            )
        ).scalar_one_or_none()

        return user

    finally:

        db.close()


# ============================================================
# POLISHED UI THEME
# ============================================================
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#f7f9fc 0%,#eef4ff 48%,#f8fbff 100%);}
.fp-hero {max-width:900px;margin:0 auto 1.2rem auto;padding:2rem 2.2rem;border:1px solid rgba(49,88,211,.12);border-radius:24px;background:rgba(255,255,255,.9);box-shadow:0 14px 40px rgba(31,55,105,.10);text-align:center;}
.fp-badge {display:inline-block;padding:.38rem .8rem;border-radius:999px;background:#eef3ff;font-weight:700;font-size:.86rem;margin-bottom:.75rem;}
.fp-title {font-size:2.45rem;line-height:1.1;font-weight:800;margin:.15rem 0 .55rem 0;}
.fp-subtitle {font-size:1.04rem;opacity:.78;margin:0 auto;max-width:680px;}
div[data-testid="stTabs"] {max-width:900px;margin-left:auto;margin-right:auto;}
div[data-testid="stTabs"] button {font-weight:650;}
div[data-testid="stTextInput"] input {border-radius:12px;}
div.stButton > button {border-radius:12px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOGIN / REGISTER
# ============================================================

if not st.session_state.logged_in:

    hero_uri = fluentpath_login_hero_data_uri()

    # ========================================================
    # FINAL CLEAN ONE-PAGE LOGIN
    # ========================================================

    st.markdown(
        f"""
        <style>

        /* ==============================================
           PAGE — DESKTOP ONE SCREEN / NO SCROLL
        ============================================== */

        html, body {{
            overflow: hidden !important;
        }}

        section.main {{
            overflow: hidden !important;
        }}

        section.main > div {{
            padding-top: 0 !important;
        }}

        .block-container {{
            max-width: 1180px !important;
            height: 100vh !important;
            padding-top: 0.45rem !important;
            padding-bottom: 0 !important;
            overflow: hidden !important;
        }}

        header[data-testid="stHeader"] {{
            height: 2rem !important;
            background: transparent !important;
        }}

        /* ==============================================
           BRAND
        ============================================== */

        .fp-one-brand {{
            height: 48px;
            display:flex;
            align-items:center;
            gap:.55rem;
            color:#12336d;
            font-size:1.45rem;
            font-weight:850;
            line-height:1;
            margin-bottom:.25rem;
        }}

        .fp-one-brand span {{
            color:#1672f8;
        }}

        .fp-one-brand small {{
            display:block;
            color:#71809a;
            font-size:.68rem;
            font-weight:600;
            margin-top:.2rem;
        }}

        /* ==============================================
           LEFT VISUAL

           Original image = 1536 x 1024.
           Only left artwork is shown.
           The printed/fake login on the original
           right side is outside this crop.
        ============================================== */

        .fp-artwork {{
            height: calc(100vh - 105px);
            min-height: 500px;
            max-height: 720px;

            border-radius:22px;
            overflow:hidden;

            background-image:url("{hero_uri}");

            /*
            Left artwork occupies about 61% of source.
            Enlarging background horizontally crops away
            the original printed login portal.
            */
            background-size: 166% 100%;
            background-position: left center;
            background-repeat:no-repeat;

            border:1px solid #dce8f7;
            box-shadow:0 14px 36px rgba(30,70,130,.09);

            position:relative;
        }}

        .fp-artwork-badge {{
            position:absolute;
            left:22px;
            bottom:18px;

            background:rgba(255,255,255,.94);
            border:1px solid rgba(215,228,245,.95);
            border-radius:14px;

            padding:.55rem .8rem;

            color:#29466f;
            font-size:.74rem;
            font-weight:750;

            box-shadow:0 6px 18px rgba(35,67,120,.08);
        }}

        /* ==============================================
           RIGHT LOGIN HEADER
        ============================================== */

        .fp-login-head {{
            text-align:center;
            margin:.05rem 0 .2rem 0;
        }}

        .fp-login-head h1 {{
            margin:0;
            color:#102f6d;
            font-size:1.55rem;
            font-weight:850;
            letter-spacing:-.025em;
        }}

        .fp-login-head h1 span {{
            color:#1672f8;
        }}

        .fp-login-head p {{
            margin:.18rem 0 0 0;
            color:#71809a;
            font-size:.78rem;
        }}

        /* ==============================================
           TABS
        ============================================== */

        div[data-testid="stTabs"] {{
            background:#ffffff;
            border:1px solid #e1eaf6;
            border-radius:16px;
            padding:.3rem .65rem .55rem .65rem;
            box-shadow:0 8px 25px rgba(35,67,120,.055);
        }}

        div[data-testid="stTabs"] button {{
            font-size:.78rem !important;
            font-weight:750 !important;
            padding-left:.45rem !important;
            padding-right:.45rem !important;
        }}

        /* ==============================================
           INPUTS
        ============================================== */

        div[data-testid="stTextInput"] {{
            margin-bottom:-.25rem !important;
        }}

        div[data-testid="stTextInput"] label {{
            color:#17366d !important;
            font-size:.78rem !important;
            font-weight:750 !important;
        }}

        div[data-testid="stTextInput"] input {{
            min-height:39px !important;
            height:39px !important;
            border-radius:9px !important;
            font-size:.82rem !important;
            background:#fbfdff !important;
        }}

        /* ==============================================
           BUTTONS
        ============================================== */

        div[data-testid="stButton"] button[kind="primary"] {{
            min-height:40px !important;
            height:40px !important;
            border-radius:9px !important;
            font-weight:800 !important;
        }}

        /* ==============================================
           SMALL TEXT
        ============================================== */

        .fp-login-message {{
            margin-top:.45rem;
            padding:.55rem .7rem;

            background:#f4f8ff;
            border:1px solid #e0eafa;
            border-radius:12px;

            text-align:center;

            color:#38547d;
            font-size:.73rem;
            font-weight:650;
        }}

        .fp-mini-title {{
            color:#15366d;
            font-size:1rem;
            font-weight:800;
            margin:.15rem 0 .05rem 0;
        }}

        .fp-mini-sub {{
            color:#71809a;
            font-size:.72rem;
            margin-bottom:.15rem;
        }}

        /* ==============================================
           REMOVE EXCESS STREAMLIT VERTICAL GAPS
        ============================================== */

        div[data-testid="stVerticalBlock"] {{
            gap:.55rem;
        }}

        div[data-testid="stAlert"] {{
            padding:.45rem .65rem !important;
            font-size:.76rem !important;
        }}

        /* ==============================================
           MOBILE FALLBACK
           Allow scrolling only on narrow screens.
        ============================================== */

        @media (max-width: 850px) {{

            html, body,
            section.main {{
                overflow:auto !important;
            }}

            .block-container {{
                height:auto !important;
                overflow:visible !important;
                padding-bottom:1rem !important;
            }}

            .fp-artwork {{
                height:360px;
                min-height:360px;
                background-size:166% 100%;
            }}

        }}

        </style>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # SMALL TOP BRAND
    # ========================================================

    st.markdown(
        """
        <div class="fp-one-brand">
            🎯
            <div>
                <span>Fluent</span>Path
                <small>AI English Coach</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # ONE PAGE — LEFT VISUAL + RIGHT REAL LOGIN
    # ========================================================

    visual_col, access_col = st.columns(
        [1.42, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # LEFT — ARTWORK ONLY
    # --------------------------------------------------------

    with visual_col:

        st.markdown(
            """
            <div class="fp-artwork">
                <div class="fp-artwork-badge">
                    Learn → Practice → Speak → Interview → Improve
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # RIGHT — ONLY REAL LOGIN
    # --------------------------------------------------------

    with access_col:

        st.markdown(
            """
            <div class="fp-login-head">
                <h1>Welcome to <span>FluentPath</span> 👋</h1>
                <p>Choose how you want to continue.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        login_tab, register_tab, admin_tab = st.tabs(
            [
                "🔐 Sign In",
                "✨ Create Account",
                "🛡️ Admin"
            ]
        )

        # ====================================================
        # SIGN IN
        # ====================================================

        with login_tab:

            st.markdown(
                """
                <div class="fp-mini-title">Welcome back</div>
                <div class="fp-mini-sub">
                    Continue your learning journey.
                </div>
                """,
                unsafe_allow_html=True
            )

            user_id = st.text_input(
                "User ID",
                value="",
                placeholder="Enter your User ID",
                key="onepage_login_user_id"
            )

            password = st.text_input(
                "Password",
                value="",
                type="password",
                placeholder="Enter your password",
                key="onepage_login_password"
            )

            if st.button(
                "Continue →",
                type="primary",
                use_container_width=True,
                key="onepage_login_button"
            ):

                if not user_id or not password:

                    st.warning(
                        "🌱 Enter your User ID and password."
                    )

                else:

                    user = authenticate_user(
                        user_id,
                        password
                    )

                    if user:

                        login_user(user)
                        st.rerun()

                    else:

                        st.info(
                            "🤝 Please check your User ID "
                            "and password and try again."
                        )

            st.markdown(
                """
                <div class="fp-login-message">
                    Better English. Brighter Opportunities.
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # CREATE ACCOUNT
        # ====================================================

        with register_tab:

            st.markdown(
                """
                <div class="fp-mini-title">
                    Start your journey ✨
                </div>
                <div class="fp-mini-sub">
                    Create your learner profile.
                </div>
                """,
                unsafe_allow_html=True
            )

            name = st.text_input(
                "Your Name",
                placeholder="Enter your name",
                key="onepage_register_name"
            )

            st.caption(
                "Your User ID will be generated automatically."
            )

            if st.button(
                "Create My Account →",
                type="primary",
                use_container_width=True,
                key="onepage_create_button"
            ):

                if not name.strip():

                    st.warning(
                        "🌱 Please enter your name."
                    )

                else:

                    user, error = create_user(name)

                    if user:

                        st.success(
                            f"🎉 Welcome, {user.name}!"
                        )

                        st.markdown(
                            f"""
                            <div class="fp-login-message">
                                YOUR USER ID<br>
                                <strong style="
                                    font-size:1.35rem;
                                    color:#1769e8;
                                ">
                                    {user.user_id}
                                </strong>
                                <br>
                                Password: <strong>2026</strong>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.warning(
                            f"🌱 We couldn’t create the account. "
                            f"{error}"
                        )

        # ====================================================
        # ADMIN
        # ====================================================

        with admin_tab:

            st.markdown(
                """
                <div class="fp-mini-title">
                    Admin access 🛡️
                </div>
                <div class="fp-mini-sub">
                    Secure administrator access.
                </div>
                """,
                unsafe_allow_html=True
            )

            admin_id = st.text_input(
                "Admin ID",
                placeholder="Enter Admin ID",
                key="onepage_admin_id"
            )

            admin_password = st.text_input(
                "Admin Password",
                type="password",
                placeholder="Enter Admin password",
                key="onepage_admin_password"
            )

            if st.button(
                "Open Admin Dashboard →",
                type="primary",
                use_container_width=True,
                key="onepage_admin_button"
            ):

                if authenticate_admin(
                    admin_id,
                    admin_password
                ):

                    st.session_state.logged_in = True
                    st.session_state.user_id = ADMIN_USER_ID
                    st.session_state.user_name = "Administrator"
                    st.session_state.is_admin = True

                    st.rerun()

                else:

                    st.warning(
                        "🌱 Admin details do not match."
                    )

            st.markdown(
                """
                <div class="fp-login-message">
                    Admin access is restricted to the
                    configured administrator.
                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# ============================================================
# STEP 12 - MULTI-USER ADMIN DASHBOARD
# ============================================================

if st.session_state.get("is_admin"):

    st.sidebar.title("🛡️ AI English Coach Admin")
    st.sidebar.success("Administrator")

    if st.sidebar.button("🚪 Admin Logout", use_container_width=True):
        logout_user()
        st.rerun()

    st.title("🛡️ Multi-User Admin Dashboard")
    st.caption(
        "Monitor registered users, learning activity, scores and individual progress."
    )

    db = SessionLocal()
    try:
        admin_users = db.execute(
            select(User).order_by(User.id.asc())
        ).scalars().all()

        admin_logs = db.execute(
            select(ActivityLog).order_by(ActivityLog.created_at.desc())
        ).scalars().all()
    finally:
        db.close()

    total_users = len(admin_users)
    total_activities = len(admin_logs)
    scored_logs = [log for log in admin_logs if log.score is not None]
    overall_avg = (
        round(sum(log.score for log in scored_logs) / len(scored_logs))
        if scored_logs else 0
    )

    today_ist = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).date()
    active_today_ids = {
        log.user_id
        for log in admin_logs
        if log.created_at
        and (log.created_at + timedelta(hours=5, minutes=30)).date() == today_ist
    }

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registered Users", total_users)
    m2.metric("Total Activities", total_activities)
    m3.metric("Overall Avg", f"{overall_avg}%" if scored_logs else "—")
    m4.metric("Active Today", len(active_today_ids))

    st.divider()
    st.subheader("👥 User Overview")

    if not admin_users:
        st.info("No registered users are available.")
    else:
        user_name_map = {
            user.user_id: user.name
            for user in admin_users
        }

        for user in admin_users:
            user_logs = [
                log for log in admin_logs
                if log.user_id == user.user_id
            ]
            user_scored = [
                log for log in user_logs
                if log.score is not None
            ]
            user_avg = (
                round(sum(log.score for log in user_scored) / len(user_scored))
                if user_scored else None
            )

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                c1.markdown(f"**{user.name}**")
                c1.caption(user.user_id)
                c2.metric("Activities", len(user_logs))
                c3.metric(
                    "Average",
                    f"{user_avg}%"
                    if user_avg is not None else "—"
                )
                c4.metric(
                    "Best",
                    f"{max(log.score for log in user_scored)}%"
                    if user_scored else "—"
                )

        st.divider()
        st.subheader("🔎 Individual User Progress")

        user_options = {
            f"{user.user_id} — {user.name}": user.user_id
            for user in admin_users
        }

        selected_label = st.selectbox(
            "Select user",
            list(user_options.keys()),
            key="step12_admin_selected_user"
        )
        selected_user_id = user_options[selected_label]

        selected_logs = [
            log for log in admin_logs
            if log.user_id == selected_user_id
        ]
        selected_scored = [
            log for log in selected_logs
            if log.score is not None
        ]

        category_scores = {}
        for category in ["Speaking", "Interview", "Test", "Assignment"]:
            values = [
                log.score for log in selected_logs
                if log.activity_type == category
                and log.score is not None
            ]
            if values:
                category_scores[category] = round(
                    sum(values) / len(values)
                )

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Saved Activities", len(selected_logs))
        p2.metric("Scored Activities", len(selected_scored))
        p3.metric(
            "Average Score",
            f"{round(sum(log.score for log in selected_scored) / len(selected_scored))}%"
            if selected_scored else "—"
        )
        p4.metric(
            "Best Score",
            f"{max(log.score for log in selected_scored)}%"
            if selected_scored else "—"
        )

        st.markdown("### 📊 Skill Performance")
        if category_scores:
            skill_cols = st.columns(len(category_scores))
            for col, (category, score) in zip(
                skill_cols,
                category_scores.items()
            ):
                col.metric(category, f"{score}%")

            strongest = max(category_scores, key=category_scores.get)
            focus = min(category_scores, key=category_scores.get)

            st.success(
                f"Strongest recorded area: **{strongest} "
                f"({category_scores[strongest]}%)**"
            )
            st.info(
                f"Current practice focus: **{focus} "
                f"({category_scores[focus]}%)**"
            )
        else:
            st.info("This user has no scored learning activity yet.")

        st.markdown("### 🕘 Recent Activity")

        if selected_logs:
            for log in selected_logs[:10]:
                if log.created_at:
                    display_time = log.created_at + timedelta(
                        hours=5,
                        minutes=30
                    )
                    time_text = display_time.strftime(
                        "%d %b %Y, %I:%M %p"
                    )
                else:
                    time_text = "Time unavailable"

                score_text = (
                    f"{int(log.score)}%"
                    if log.score is not None
                    else "Completed"
                )

                with st.container(border=True):
                    r1, r2 = st.columns([4, 1])
                    r1.markdown(
                        f"**{log.activity_type}** — {log.activity}"
                    )
                    r1.caption(time_text)
                    r2.markdown(f"**{score_text}**")
        else:
            st.info("This learning journey is ready to begin. The first completed activity will appear here.")

    # ========================================================
    # ADMIN USER MANAGEMENT - PERMANENT DELETE
    # ========================================================

    st.divider()
    st.markdown("### ⚠️ Permanent User Management")

    st.warning(
        "Deleting a learner permanently removes the learner account, "
        "activity history and saved review data. This action cannot be undone."
    )

    if not admin_users:
        st.info("There are no learner accounts available to manage.")

    else:
        delete_user_options = {
            f"{user.user_id} — {user.name}": user.user_id
            for user in admin_users
        }

        delete_label = st.selectbox(
            "Select learner to manage",
            list(delete_user_options.keys()),
            key="admin_delete_selected_user"
        )

        delete_user_id = delete_user_options[delete_label]

        delete_user_obj = next(
            (
                user for user in admin_users
                if user.user_id == delete_user_id
            ),
            None
        )

        delete_logs = [
            log for log in admin_logs
            if log.user_id == delete_user_id
        ]

        db_preview = SessionLocal()

        try:
            delete_review_count = (
                db_preview.query(ActivityReview)
                .filter(
                    ActivityReview.user_id == delete_user_id
                )
                .count()
            )
        finally:
            db_preview.close()

        with st.container(border=True):

            st.markdown(
                f"**Learner:** "
                f"{delete_user_obj.name if delete_user_obj else 'Unknown'}"
            )

            st.write(f"**User ID:** `{delete_user_id}`")

            d1, d2 = st.columns(2)

            d1.metric(
                "Activity Records",
                len(delete_logs)
            )

            d2.metric(
                "Saved Reviews",
                delete_review_count
            )

            st.caption(
                "Both activity records and saved review data will be "
                "removed together with this learner account."
            )

        expected_delete_text = f"DELETE {delete_user_id}"

        st.markdown(
            f"To confirm permanent deletion, type exactly: "
            f"**`{expected_delete_text}`**"
        )

        delete_confirmation = st.text_input(
            "Deletion confirmation",
            key=f"admin_delete_confirmation_{delete_user_id}",
            placeholder=expected_delete_text
        )

        confirmation_ok = (
            delete_confirmation.strip() == expected_delete_text
        )

        if delete_confirmation and not confirmation_ok:
            st.info(
                f"Confirmation does not match. Type exactly: "
                f"{expected_delete_text}"
            )

        delete_clicked = st.button(
            "🗑️ Permanently Delete Learner",
            type="primary",
            use_container_width=True,
            disabled=not confirmation_ok,
            key=f"admin_permanent_delete_{delete_user_id}"
        )

        if delete_clicked:

            # Extra protection: configured administrator must never
            # be handled as a learner deletion target.
            if delete_user_id == ADMIN_USER_ID:
                st.error(
                    "The administrator account cannot be deleted here."
                )

            else:
                db_delete = SessionLocal()

                try:
                    target_user = db_delete.execute(
                        select(User).where(
                            User.user_id == delete_user_id
                        )
                    ).scalar_one_or_none()

                    if target_user is None:
                        st.warning(
                            "This learner account no longer exists."
                        )

                    else:
                        # 1. Delete full review payloads first.
                        reviews_deleted = (
                            db_delete.query(ActivityReview)
                            .filter(
                                ActivityReview.user_id
                                == delete_user_id
                            )
                            .delete(
                                synchronize_session=False
                            )
                        )

                        # 2. Delete learning activity history.
                        logs_deleted = (
                            db_delete.query(ActivityLog)
                            .filter(
                                ActivityLog.user_id
                                == delete_user_id
                            )
                            .delete(
                                synchronize_session=False
                            )
                        )

                        # 3. Delete learner account last.
                        db_delete.delete(target_user)

                        # One transaction for all three operations.
                        db_delete.commit()

                        st.session_state[
                            "admin_delete_success"
                        ] = (
                            f"{delete_user_id} deleted successfully. "
                            f"Removed {logs_deleted} activity record(s) "
                            f"and {reviews_deleted} saved review(s)."
                        )

                        st.rerun()

                except Exception as e:
                    db_delete.rollback()

                    st.error(
                        "Deletion was cancelled because the database "
                        f"operation failed. No partial delete was committed. "
                        f"Details: {e}"
                    )

                finally:
                    db_delete.close()

    delete_success_message = st.session_state.pop(
        "admin_delete_success",
        None
    )

    if delete_success_message:
        st.success(
            f"✅ {delete_success_message}"
        )

    st.divider()
    st.subheader("📌 Admin Notes")
    st.write(
        "The dashboard monitors learner progress. Permanent deletion is "
        "restricted to the protected User Management section and requires "
        "exact confirmation before it can run."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================


# ============================================================
# STEP 17B - FLUENTPATH FULL VISUAL SYSTEM
# Applies consistently from Dashboard through History.
# Business logic, database logic, AI logic and WebRTC are unchanged.
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- App canvas ---------- */
    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(222,235,255,.55), transparent 28%),
            linear-gradient(180deg, #f8fbff 0%, #ffffff 42%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }

    /* ---------- Typography ---------- */
    h1 {
        color: #14213d !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
        margin-bottom: .35rem !important;
    }
    h2, h3 {
        color: #1d2d50 !important;
        font-weight: 750 !important;
        letter-spacing: -0.015em;
    }
    p, li, label {
        line-height: 1.55;
    }

    /* ---------- Main cards / bordered containers ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #e5edf8 !important;
        border-radius: 18px !important;
        background: rgba(255,255,255,.92) !important;
        box-shadow: 0 8px 26px rgba(35,67,120,.055);
    }

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f5f9ff);
        border: 1px solid #e4edf9;
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: 0 8px 24px rgba(41,72,120,.06);
        min-height: 112px;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 650;
        color: #52627a;
    }
    [data-testid="stMetricValue"] {
        color: #172554;
        font-weight: 800;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 12px !important;
        min-height: 2.65rem;
        font-weight: 700 !important;
        border: 1px solid #d9e5f7 !important;
        box-shadow: 0 4px 12px rgba(35,67,120,.045);
        transition: transform .12s ease, box-shadow .12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(35,67,120,.09);
    }

    /* ---------- Inputs ---------- */
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div {
        border-radius: 12px !important;
    }
    textarea {
        border-radius: 12px !important;
    }

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border-width: 1px !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: .35rem;
        border-bottom: 1px solid #e7edf6;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding-left: .85rem;
        padding-right: .85rem;
        font-weight: 650;
    }

    /* ---------- Progress ---------- */
    .stProgress > div > div > div > div {
        border-radius: 999px;
    }

    /* ---------- Dividers ---------- */
    hr {
        border-color: #edf2f8 !important;
        margin: 1.35rem 0 !important;
    }

    /* ---------- FluentPath custom components ---------- */
    .fp-hero {
        background: linear-gradient(135deg, #f2f7ff 0%, #ffffff 52%, #eefbf5 100%);
        border: 1px solid #e1eaf7;
        border-radius: 24px;
        padding: 1.35rem 1.5rem;
        margin: .15rem 0 1.25rem 0;
        box-shadow: 0 12px 32px rgba(38,72,122,.07);
    }
    .fp-hero-title {
        color: #14213d;
        font-size: 2rem;
        line-height: 1.15;
        font-weight: 850;
        letter-spacing: -.03em;
        margin: 0 0 .4rem 0;
    }
    .fp-hero-sub {
        color: #64748b;
        font-size: 1rem;
        margin: 0;
    }
    .fp-pill {
        display: inline-block;
        padding: .34rem .7rem;
        margin-top: .8rem;
        border-radius: 999px;
        background: #eef5ff;
        color: #2856a6;
        font-size: .78rem;
        font-weight: 750;
    }
    .fp-section-label {
        color: #64748b;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin: .25rem 0 .3rem 0;
    }
    .fp-quote {
        background: linear-gradient(135deg, #f7faff, #f2f8ff);
        border: 1px solid #e2eaf6;
        border-radius: 20px;
        padding: 1.2rem 1.35rem;
        color: #334155;
        font-size: 1rem;
        margin-top: 1rem;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fbfdff 0%, #f7faff 100%);
        border-right: 1px solid #e5edf8;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        transition: background .12s ease, transform .12s ease;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #eef5ff;
        transform: translateX(2px);
    }

    /* ---------- Mobile ---------- */
    @media (max-width: 768px) {
        .block-container {
            padding-top: .8rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .fp-hero {
            padding: 1rem;
            border-radius: 18px;
        }
        .fp-hero-title {
            font-size: 1.55rem;
        }
        [data-testid="stMetric"] {
            min-height: 96px;
            padding: .8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# STEP 17A - POLISHED FLUENTPATH SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 270px;
        max-width: 270px;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.25rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        min-height: 2.25rem;
        padding: 0.25rem 0.65rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.05rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 9px;
        padding-top: 0.22rem;
        padding-bottom: 0.22rem;
        margin-bottom: 0.02rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="padding:0.15rem 0 0.35rem 0;">
        <div style="font-size:1.42rem;font-weight:800;line-height:1.1;">🎯 FluentPath</div>
        <div style="font-size:0.78rem;opacity:0.68;margin-top:0.15rem;">AI English Coach</div>
    </div>
    """,
    unsafe_allow_html=True
)

profile_col, logout_col = st.sidebar.columns([1.75, 1.05], gap="small")

with profile_col:
    st.markdown(
        f"""
        <div style="line-height:1.2;padding:0.20rem 0;">
            <div style="font-weight:700;font-size:0.96rem;">👤 {st.session_state.user_name}</div>
            <div style="font-size:0.76rem;opacity:0.68;">ID: {st.session_state.user_id}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with logout_col:
    if st.button(
        "Logout",
        key="step17a_polished_logout",
        use_container_width=True
    ):
        logout_user()
        st.rerun()

st.sidebar.markdown(
    '<div style="font-size:0.78rem;font-weight:700;opacity:0.60;'
    'margin:0.35rem 0 0.15rem 0;">NAVIGATION</div>',
    unsafe_allow_html=True
)

polished_page = st.sidebar.radio(
    "Navigation",
    [
        "🏠  Dashboard",
        "📖  Learning",
        "🎯  Practice",
        "☑️  Tests",
        "📅  Daily Assignment",
        "🎙️  Speaking",
        "👥  Interviews",
        "🧑‍💻  TechPrep",
        "💬  SmartSpeak",
        "🧠  AI Architecture",
        "📊  Progress",
        "🕘  History"
    ],
    label_visibility="collapsed"
)

STEP17A_PAGE_MAP = {
    "🏠  Dashboard": "🏠 Dashboard",
    "📖  Learning": "📚 Learning",
    "🎯  Practice": "📝 Practice",
    "☑️  Tests": "🧪 Tests",
    "📅  Daily Assignment": "📝 Daily Assignment",
    "🎙️  Speaking": "🗣️ Speaking",
    "👥  Interviews": "🎤 Interviews",
    "🧑‍💻  TechPrep": "🧑‍💻 TechPrep",
    "💬  SmartSpeak": "💬 SmartSpeak",
    "🧠  AI Architecture": "🧠 AI Architecture",
    "📊  Progress": "📈 Progress",
    "🕘  History": "📜 History",
}

page = STEP17A_PAGE_MAP[polished_page]


# ============================================================
# LEARNING CENTER — BUILT-IN MODULE NAVIGATION
# ============================================================

if page == "📚 Learning":

    st.markdown(
        """
        <style>
        div[data-testid="stSegmentedControl"] {
            margin-bottom: .55rem;
        }

        div[data-testid="stSegmentedControl"] button {
            font-size: .82rem !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    learning_section = st.segmented_control(
        "Learning Modules",
        [
            "🏠 Learning Home",
            "📘 Grammar",
            "⏱️ Tenses",
            "🧠 Vocabulary",
            "✍️ Sentence Formation"
        ],
        default="🏠 Learning Home",
        selection_mode="single",
        key="learning_center_module"
    )

    LEARNING_PAGE_MAP = {
        "🏠 Learning Home": "📚 Learning",
        "📘 Grammar": "📘 Grammar",
        "⏱️ Tenses": "⏱️ Tenses",
        "🧠 Vocabulary": "🧠 Vocabulary",
        "✍️ Sentence Formation": "✍️ Sentence Formation",
    }

    if learning_section:
        page = LEARNING_PAGE_MAP[learning_section]


# ============================================================
# STEP 8 - PERSONALIZED DAILY LEARNING PLAN
# ============================================================

if page == "🏠 Dashboard":

    friendly_coach_message("🏠 Dashboard")

    dashboard_date = datetime.now().strftime("%a, %d %b %Y")
    st.markdown(
        f"""
        <div class="fp-hero">
            <div class="fp-section-label">Personal English Coach</div>
            <div class="fp-hero-title">Welcome back, {st.session_state.user_name}! 👋</div>
            <p class="fp-hero-sub">
                Keep learning, keep improving. You’re on the right path.
            </p>
            <span class="fp-pill">📅 {dashboard_date} &nbsp; • &nbsp; ID: {st.session_state.user_id}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    db = SessionLocal()
    try:
        dashboard_logs = (
            db.query(ActivityLog)
            .filter(ActivityLog.user_id == st.session_state.user_id)
            .order_by(ActivityLog.created_at.desc())
            .all()
        )
    finally:
        db.close()

    scored_dashboard = [
        log for log in dashboard_logs
        if log.score is not None
    ]

    def dashboard_average(values):
        return round(sum(values) / len(values)) if values else 0

    speaking_values = [
        log.score for log in scored_dashboard
        if log.activity_type == "Speaking"
    ]
    interview_values = [
        log.score for log in scored_dashboard
        if log.activity_type == "Interview"
    ]
    test_values = [
        log.score for log in scored_dashboard
        if log.activity_type == "Test"
    ]

    st.divider()
    st.markdown("### 📊 Your Learning Snapshot")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "🗣️ Speaking",
        f"{dashboard_average(speaking_values)}%"
        if speaking_values else "—"
    )
    d2.metric(
        "🎤 Interview",
        f"{dashboard_average(interview_values)}%"
        if interview_values else "—"
    )
    d3.metric(
        "🧪 Tests",
        f"{dashboard_average(test_values)}%"
        if test_values else "—"
    )

    combined_values = (
        speaking_values + interview_values + test_values
    )

    d4.metric(
        "⭐ Overall",
        f"{dashboard_average(combined_values)}%"
        if combined_values else "—"
    )

    category_scores = {}
    if speaking_values:
        category_scores["Speaking"] = dashboard_average(speaking_values)
    if interview_values:
        category_scores["Interview"] = dashboard_average(interview_values)
    if test_values:
        category_scores["Tests"] = dashboard_average(test_values)

    if category_scores:
        focus_area = min(category_scores, key=category_scores.get)
        focus_score = category_scores[focus_area]
    else:
        focus_area = "Speaking"
        focus_score = None

    st.divider()
    st.markdown("### 🎯 Practice Today for a Better Tomorrow")

    today_text = datetime.now().strftime("%d %B %Y")
    st.caption(f"Plan for {today_text}")

    if focus_score is not None:
        st.info(
            f"Your current priority is **{focus_area}** "
            f"({focus_score}%). Today's plan gives extra attention to this area."
        )
    else:
        st.info(
            "Start with the basic daily plan. Your coach will personalize it "
            "after you complete scored activities."
        )

    plan_items = []

    if focus_area == "Interview":
        plan_items = [
            (
                "1️⃣ Interview Focus",
                "Practice one 5-question interview. Use: direct answer → "
                "reason → example → result."
            ),
            (
                "2️⃣ Speaking Support",
                "Answer one Guided Speaking question and retry once using "
                "the Adaptive Coach tip."
            ),
            (
                "3️⃣ Grammar Review",
                "Review one grammar topic and write 3 simple example sentences."
            ),
            (
                "4️⃣ Vocabulary",
                "Learn 5 useful professional words and use each in a sentence."
            ),
        ]
    elif focus_area == "Speaking":
        plan_items = [
            (
                "1️⃣ Speaking Focus",
                "Complete one Guided Speaking question, check the AI score, "
                "use the coach tip and retry the same question."
            ),
            (
                "2️⃣ Voice Practice",
                "Record one short answer and check transcript, fluency and "
                "voice accuracy."
            ),
            (
                "3️⃣ Sentence Formation",
                "Practice: main point → reason → example."
            ),
            (
                "4️⃣ Interview Support",
                "Answer one HR interview question in under 2 minutes."
            ),
        ]
    else:
        plan_items = [
            (
                "1️⃣ Test Focus",
                "Review one learning topic and complete a short test."
            ),
            (
                "2️⃣ Grammar",
                "Review the mistakes from your latest practice."
            ),
            (
                "3️⃣ Speaking",
                "Answer one Guided Speaking question naturally."
            ),
            (
                "4️⃣ Interview",
                "Practice one interview question using a clear structure."
            ),
        ]

    for title, task in plan_items:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(task)

    st.divider()
    st.markdown("### 🧠 Coach Direction")

    if focus_score is None:
        st.write(
            "Complete today's activities first. Your saved scores will be used "
            "to identify your next focus area."
        )
    elif focus_score < 50:
        st.warning(
            f"Focus on **{focus_area}** with short and simple answers. "
            "Do not try to make answers long. First improve structure and clarity."
        )
    elif focus_score < 70:
        st.info(
            f"Your **{focus_area}** is developing. Use examples, review the "
            "coach feedback, and retry after every important practice."
        )
    else:
        st.success(
            f"Your **{focus_area}** is progressing well. Keep it consistent "
            "while improving the other areas."
        )

    st.divider()
    st.markdown("### 🔥 Your Learning Progress")

    activity_dates = set()

    for log in dashboard_logs:
        if log.created_at is not None:
            ist_time = log.created_at + timedelta(hours=5, minutes=30)
            activity_dates.add(ist_time.date())

    today_date = datetime.now().date()
    streak = 0
    check_date = today_date

    # If there is no activity today, allow the streak to continue from yesterday.
    if check_date not in activity_dates:
        check_date = check_date - timedelta(days=1)

    while check_date in activity_dates:
        streak += 1
        check_date = check_date - timedelta(days=1)

    s1, s2 = st.columns(2)
    s1.metric("Current Streak", f"{streak} day" if streak == 1 else f"{streak} days")
    s2.metric("Total Saved Activities", len(dashboard_logs))

    if streak >= 7:
        st.success("🏆 Excellent consistency — one week or more of active learning.")
    elif streak >= 3:
        st.success("🔥 Good consistency. Keep the learning streak going.")
    elif streak >= 1:
        st.info("Keep practicing daily to build your learning streak.")
    else:
        st.info("Complete one activity today to start your learning streak.")

    st.divider()
    st.markdown("### ⚡ Your Learning Loop")
    st.write(
        "Learn → Practice → AI Feedback → Detect Weakness → "
        "Personalized Practice → Retry → Track Improvement"
    )

    st.markdown(
        """
        <div class="fp-quote">
            <strong>“Small steps make big progress.”</strong><br>
            Communication grows through consistent practice, clear feedback,
            and the confidence to try again.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# LEARNING
# ============================================================

    # ========================================================
    # LEARNING MEMORY DASHBOARD V1
    # ========================================================
    #
    # Persistent learner state reconstructed from ActivityLog.
    #
    # Read-only:
    # - no DB schema change
    # - no new table
    # - no LLM/API call
    #
    # ========================================================

    try:

        _memory_user_id = (
            st.session_state.get(
                "user_id"
            )
        )

        if _memory_user_id:

            _learning_memory = (
                build_learning_memory(
                    _memory_user_id
                )
            )

            st.markdown("---")

            st.subheader(
                "🧠 Learning Memory"
            )

            st.caption(
                "Your recent learning history is used "
                "to identify progress and recommend "
                "the next useful practice."
            )


            # ------------------------------------------------
            # ACTIVE LEARNER
            # ------------------------------------------------

            if (
                _learning_memory.get(
                    "status"
                )
                == "ACTIVE"
            ):

                _lm1, _lm2, _lm3, _lm4 = (
                    st.columns(4)
                )

                _lm1.metric(
                    "Current Level",
                    str(
                        _learning_memory.get(
                            "overall_level",
                            "New Learner"
                        )
                    )
                )

                _memory_average = (
                    _learning_memory.get(
                        "overall_average"
                    )
                )

                _lm2.metric(
                    "Average Score",
                    (
                        str(
                            _memory_average
                        )
                        + "%"
                        if _memory_average
                        is not None
                        else "—"
                    )
                )

                _lm3.metric(
                    "Trend",
                    str(
                        _learning_memory.get(
                            "trend",
                            "Not Enough Data"
                        )
                    )
                )

                _lm4.metric(
                    "Scored Practice",
                    str(
                        _learning_memory.get(
                            "scored_activities",
                            0
                        )
                    )
                )


                # --------------------------------------------
                # STRONGEST + FOCUS AREA
                # --------------------------------------------

                _strongest = (
                    _learning_memory.get(
                        "strongest_topic"
                    )
                )

                _weakest = (
                    _learning_memory.get(
                        "weakest_topic"
                    )
                )

                _la, _lb = st.columns(2)


                with _la:

                    st.markdown(
                        "#### 💪 Strongest Area"
                    )

                    if _strongest:

                        st.write(
                            "**"
                            + str(
                                _strongest.get(
                                    "module",
                                    "General"
                                )
                            )
                            + "**"
                        )

                        st.write(
                            str(
                                _strongest.get(
                                    "topic",
                                    "General"
                                )
                            )
                        )

                        _strong_avg = (
                            _strongest.get(
                                "average_score"
                            )
                        )

                        if (
                            _strong_avg
                            is not None
                        ):

                            st.caption(
                                "Average: "
                                + str(
                                    _strong_avg
                                )
                                + "%"
                            )

                        st.caption(
                            "Level: "
                            + str(
                                _strongest.get(
                                    "level",
                                    "Beginner"
                                )
                            )
                            + " • Trend: "
                            + str(
                                _strongest.get(
                                    "trend",
                                    "Not Enough Data"
                                )
                            )
                        )

                    else:

                        st.write(
                            "Complete scored practice "
                            "to identify your strongest area."
                        )


                with _lb:

                    st.markdown(
                        "#### 🎯 Focus Area"
                    )

                    if _weakest:

                        st.write(
                            "**"
                            + str(
                                _weakest.get(
                                    "module",
                                    "General"
                                )
                            )
                            + "**"
                        )

                        st.write(
                            str(
                                _weakest.get(
                                    "topic",
                                    "General"
                                )
                            )
                        )

                        _weak_avg = (
                            _weakest.get(
                                "average_score"
                            )
                        )

                        if (
                            _weak_avg
                            is not None
                        ):

                            st.caption(
                                "Average: "
                                + str(
                                    _weak_avg
                                )
                                + "%"
                            )

                        st.caption(
                            "Level: "
                            + str(
                                _weakest.get(
                                    "level",
                                    "Beginner"
                                )
                            )
                            + " • Trend: "
                            + str(
                                _weakest.get(
                                    "trend",
                                    "Not Enough Data"
                                )
                            )
                        )

                    else:

                        st.write(
                            "No focus area identified yet."
                        )


                # --------------------------------------------
                # RECOMMENDATION
                # --------------------------------------------

                _recommendation = (
                    _learning_memory.get(
                        "recommendation",
                        {}
                    )
                )

                st.markdown(
                    "#### 🚀 Recommended Next Practice"
                )

                _recommended_module = (
                    _recommendation.get(
                        "recommended_module",
                        "Practice"
                    )
                )

                _recommended_topic = (
                    _recommendation.get(
                        "recommended_topic",
                        "General"
                    )
                )

                _recommended_difficulty = (
                    _recommendation.get(
                        "recommended_difficulty",
                        "Beginner"
                    )
                )

                st.success(
                    str(
                        _recommended_module
                    )
                    + " → "
                    + str(
                        _recommended_topic
                    )
                    + " → "
                    + str(
                        _recommended_difficulty
                    )
                )

                _recommendation_reason = (
                    _recommendation.get(
                        "reason",
                        ""
                    )
                )

                if _recommendation_reason:

                    st.caption(
                        str(
                            _recommendation_reason
                        )
                    )


                # --------------------------------------------
                # LAST LEARNING ACTIVITY
                # --------------------------------------------

                st.markdown(
                    "#### 🕘 Continue From"
                )

                st.write(
                    "**Last Module:** "
                    + str(
                        _learning_memory.get(
                            "latest_module",
                            "Not Available"
                        )
                    )
                )

                st.write(
                    "**Last Topic:** "
                    + str(
                        _learning_memory.get(
                            "latest_topic",
                            "Not Available"
                        )
                    )
                )

                st.caption(
                    "Last activity: "
                    + str(
                        _learning_memory.get(
                            "last_activity",
                            "Not Available"
                        )
                    )
                )


            # ------------------------------------------------
            # NEW LEARNER
            # ------------------------------------------------

            elif (
                _learning_memory.get(
                    "status"
                )
                == "NEW_LEARNER"
            ):

                st.info(
                    "Your learning memory will build "
                    "automatically as you complete "
                    "practice activities."
                )

                _new_recommendation = (
                    _learning_memory.get(
                        "recommendation",
                        {}
                    )
                )

                st.markdown(
                    "#### 🚀 Recommended First Practice"
                )

                st.success(
                    str(
                        _new_recommendation.get(
                            "recommended_module",
                            "TechPrep"
                        )
                    )
                    + " → "
                    + str(
                        _new_recommendation.get(
                            "recommended_topic",
                            "Python"
                        )
                    )
                    + " → "
                    + str(
                        _new_recommendation.get(
                            "recommended_difficulty",
                            "Beginner"
                        )
                    )
                )


            # ------------------------------------------------
            # SAFE UNKNOWN STATE
            # ------------------------------------------------

            else:

                st.info(
                    "Learning Memory will become "
                    "available after your learning "
                    "history is recorded."
                )


    except Exception as _learning_memory_error:

        # Memory must never break Dashboard.
        st.caption(
            "🧠 Learning Memory is temporarily "
            "unavailable. Your existing learning "
            "features are not affected."
        )

elif page == "📚 Learning":

    friendly_coach_message("📚 Learning")

    st.title("📚 Learning Center")
    st.write(
        "Build your English foundation before practice. "
        "Use Grammar, Tenses, Vocabulary and Sentence Formation from the sidebar."
    )

    st.subheader("🧭 Recommended Learning Order")
    st.write(
        "1. Grammar → 2. Tenses → 3. Vocabulary → "
        "4. Sentence Formation → 5. Practice → 6. Test"
    )

    st.subheader("🎯 How to use this section")
    st.info(
        "Study one small topic, read the examples, create your own examples, "
        "then move to Practice or Tests. Keep the learning simple and regular."
    )


# ============================================================
# GRAMMAR
# ============================================================

elif page == "📘 Grammar":

    friendly_coach_message("📘 Grammar")

    st.title("📖 Grammar Learning")

    grammar_topics = {
        "Parts of Speech": {
            "idea": "Words have different jobs in a sentence.",
            "points": [
                "Noun — person, place, thing or idea.",
                "Pronoun — replaces a noun.",
                "Verb — shows an action or state.",
                "Adjective — describes a noun.",
                "Adverb — describes a verb, adjective or another adverb.",
                "Preposition — shows relationship or position.",
                "Conjunction — connects words or ideas."
            ],
            "examples": [
                "The engineer checked the site.",
                "She explained the problem clearly.",
                "The team worked carefully and completed the task."
            ]
        },
        "Articles": {
            "idea": "Use a, an and the before nouns when appropriate.",
            "points": [
                "Use 'a' before a consonant sound: a project.",
                "Use 'an' before a vowel sound: an engineer.",
                "Use 'the' for something specific: the report."
            ],
            "examples": [
                "I am an engineer.",
                "We started a new project.",
                "The meeting starts at 10 AM."
            ]
        },
        "Subject–Verb Agreement": {
            "idea": "The verb must agree with the subject.",
            "points": [
                "I/You/We/They work.",
                "He/She/It works.",
                "A singular subject usually takes a singular verb.",
                "A plural subject usually takes a plural verb."
            ],
            "examples": [
                "She works in technical services.",
                "They visit sites regularly.",
                "The customer needs technical support."
            ]
        },
        "Prepositions": {
            "idea": "Prepositions connect nouns or pronouns with other parts of a sentence.",
            "points": [
                "Use 'at' for a specific time: at 9 AM.",
                "Use 'on' for days/dates: on Monday.",
                "Use 'in' for months, years and longer periods: in September.",
                "Common place words include at, in, on, near and between."
            ],
            "examples": [
                "The meeting is at 10 AM.",
                "I will visit the site on Monday.",
                "I joined the course in August."
            ]
        },
        "Conjunctions": {
            "idea": "Conjunctions connect words, phrases or ideas.",
            "points": [
                "and — adds information.",
                "but — shows contrast.",
                "because — gives a reason.",
                "so — shows a result."
            ],
            "examples": [
                "I studied the data and prepared the report.",
                "The task was difficult, but we completed it.",
                "I practiced daily because I wanted to improve."
            ]
        },
        "Question Formation": {
            "idea": "English questions often use helping verbs before the subject.",
            "points": [
                "Do/Does for simple present questions.",
                "Did for simple past questions.",
                "Is/Are for present continuous or states.",
                "Can/Will for ability or future questions."
            ],
            "examples": [
                "Do you work here?",
                "Did you complete the report?",
                "Are you learning English?",
                "Can you explain the problem?"
            ]
        }
    }

    selected_grammar = st.selectbox(
        "Choose grammar topic",
        list(grammar_topics.keys()),
        key="step11_grammar_topic"
    )

    g = grammar_topics[selected_grammar]
    st.subheader(selected_grammar)
    st.info(g["idea"])

    st.markdown("### Key Points")
    for point in g["points"]:
        st.write(f"• {point}")

    st.markdown("### Examples")
    for example in g["examples"]:
        st.write(f"✅ {example}")

    st.info(
        "📘 Learn this topic here. When you are ready, go to 🎯 Practice for guided practice, "
        "☑️ Tests to check your understanding, and 🎙️ Speaking to practise it aloud."
    )


# ============================================================
# TENSES
# ============================================================

elif page == "⏱️ Tenses":

    friendly_coach_message("⏱️ Tenses")

    st.title("⏳ 12 English Tenses")

    tense_data = {
        "1. Simple Present": (
            "Subject + base verb / verb+s",
            "Routine, fact or regular action.",
            "I visit sites regularly."
        ),
        "2. Present Continuous": (
            "Subject + am/is/are + verb-ing",
            "Action happening now or around now.",
            "I am learning English now."
        ),
        "3. Present Perfect": (
            "Subject + have/has + past participle",
            "Completed action connected to the present.",
            "I have completed the report."
        ),
        "4. Present Perfect Continuous": (
            "Subject + have/has been + verb-ing",
            "Action continuing from the past until now.",
            "I have been learning English for two months."
        ),
        "5. Simple Past": (
            "Subject + past form",
            "Completed action in the past.",
            "I visited the site yesterday."
        ),
        "6. Past Continuous": (
            "Subject + was/were + verb-ing",
            "Action in progress at a past time.",
            "I was working at 8 PM."
        ),
        "7. Past Perfect": (
            "Subject + had + past participle",
            "Action completed before another past action.",
            "I had finished the work before the meeting started."
        ),
        "8. Past Perfect Continuous": (
            "Subject + had been + verb-ing",
            "Ongoing action before another past point.",
            "I had been working for two hours before lunch."
        ),
        "9. Simple Future": (
            "Subject + will + base verb",
            "Future action, decision or prediction.",
            "I will attend the meeting tomorrow."
        ),
        "10. Future Continuous": (
            "Subject + will be + verb-ing",
            "Action that will be in progress at a future time.",
            "I will be travelling at 9 AM."
        ),
        "11. Future Perfect": (
            "Subject + will have + past participle",
            "Action completed before a future time.",
            "I will have completed the report by 6 PM."
        ),
        "12. Future Perfect Continuous": (
            "Subject + will have been + verb-ing",
            "Duration of an action up to a future point.",
            "By next month, I will have been learning English for three months."
        ),
    }

    tense = st.selectbox(
        "Choose a tense",
        list(tense_data.keys()),
        key="step11_tense_topic"
    )

    formula, use, example = tense_data[tense]

    st.markdown(f"### {tense}")
    st.write(f"**Formula:** {formula}")
    st.write(f"**Use:** {use}")
    st.success(f"Example: {example}")

    st.markdown("### 🧠 Remember")
    st.write(
        "First identify the time of the action. Then choose the tense and "
        "build the sentence using the correct helping verb and verb form."
    )

    st.info(
        "🕒 Learn the tense, formula and examples here. Then go to 🎯 Practice to apply it, "
        "☑️ Tests to check your tense knowledge, and 🎙️ Speaking to use it naturally."
    )


# ============================================================
# VOCABULARY
# ============================================================

elif page == "🧠 Vocabulary":

    friendly_coach_message("🧠 Vocabulary")

    st.title("🧾 Vocabulary Builder")

    vocab_groups = {
        "Workplace": [
            ("coordinate", "organize people or activities", "I coordinate with the site team."),
            ("priority", "something that needs attention first", "Safety is our first priority."),
            ("deadline", "the final time for completing something", "We completed the task before the deadline."),
            ("feedback", "comments used to improve something", "The customer gave positive feedback."),
            ("responsibility", "a duty you are expected to handle", "Customer support is one of my responsibilities.")
        ],
        "Communication": [
            ("clarify", "make something easier to understand", "I clarified the requirement with the customer."),
            ("explain", "make an idea clear by giving details", "I explained the process to the team."),
            ("respond", "reply or react", "I responded to the customer's question."),
            ("discuss", "talk about a topic with others", "We discussed the project issue."),
            ("confirm", "state that something is correct or certain", "Please confirm the meeting time.")
        ],
        "Problem Solving": [
            ("identify", "find or recognize something", "We identified the root cause."),
            ("analyze", "study something carefully", "I analyzed the sales data."),
            ("solution", "a way to solve a problem", "We found a practical solution."),
            ("improve", "make something better", "I want to improve my communication."),
            ("resolve", "solve a problem or disagreement", "The team resolved the issue quickly.")
        ],
        "Interview": [
            ("achievement", "something successfully completed", "One achievement was completing the project before the deadline."),
            ("strength", "a quality or skill you do well", "Problem solving is one of my strengths."),
            ("experience", "knowledge gained by doing work", "I have practical industry experience."),
            ("adapt", "change effectively for a new situation", "I can adapt to new responsibilities."),
            ("collaborate", "work together with others", "I collaborate with engineers and customers.")
        ]
    }

    group = st.selectbox(
        "Choose vocabulary category",
        list(vocab_groups.keys()),
        key="step11_vocab_group"
    )

    st.subheader(f"📘 {group} Words")

    for word, meaning, example in vocab_groups[group]:
        with st.container(border=True):
            st.markdown(f"### {word}")
            st.write(f"**Meaning:** {meaning}")
            st.write(f"**Example:** {example}")

    st.info(
        "📚 Learn the words, meanings and examples here. Then go to 🎯 Practice to use them, "
        "☑️ Tests to check your vocabulary, and 🎙️ Speaking to use the new words naturally."
    )


# ============================================================
# SENTENCE FORMATION
# ============================================================

elif page == "✍️ Sentence Formation":

    friendly_coach_message("✍️ Sentence Formation")

    st.title("🧩 Sentence Formation")

    st.subheader("1️⃣ Basic Sentence")
    st.write("**Subject + Verb + Object**")
    st.success("I prepared the report.")

    st.subheader("2️⃣ Add a Reason")
    st.write("**Main point + because + reason**")
    st.success("I joined the course because I wanted to learn AI.")

    st.subheader("3️⃣ Add an Example")
    st.write("**Main point → reason → example**")
    st.success(
        "I enjoy solving practical problems. It helps me learn from real situations. "
        "For example, I use data to understand sales performance."
    )

    st.subheader("4️⃣ Professional Answer Structure")
    st.write("**Direct answer → reason → example → result**")
    st.success(
        "I work well with customers because I listen to their requirements carefully. "
        "For example, I visit sites and understand technical issues before suggesting "
        "a solution. This helps me provide more practical support."
    )

    st.subheader("5️⃣ Behavioral Interview Structure")
    st.write("**Situation → Task → Action → Result (STAR)**")
    st.info(
        "Situation: What happened?  |  Task: What did you need to do?  |  "
        "Action: What did you do?  |  Result: What happened finally?"
    )

    st.divider()
    st.subheader("🔧 Common Sentence Fixes")

    fixes = [
        ("I am working here since two years.", "I have been working here for two years."),
        ("She go to office every day.", "She goes to the office every day."),
        ("I am completed the work.", "I have completed the work."),
        ("I discussed about the issue.", "I discussed the issue."),
        ("I have 11 years experience.", "I have 11 years of experience.")
    ]

    for wrong, correct in fixes:
        st.write(f"❌ {wrong}")
        st.write(f"✅ {correct}")
        st.write("")

    st.info(
        "✏️ Learn the sentence structures and examples here. Then go to 🎯 Practice to build answers, "
        "☑️ Tests to check sentence formation, and 🎙️ Speaking or 👥 Interviews to practise aloud."
    )


# ============================================================
# PRACTICE
# ============================================================



elif page == "📝 Practice":

    friendly_coach_message("📝 Practice")

    st.title("📝 English Practice")

    practice_data = {

        "Grammar": [
            (
                "Choose the correct sentence.",
                ["She works every day.", "She work every day."],
                0
            ),
            (
                "Choose the correct article.",
                ["I bought a book.", "I bought book."],
                0
            ),
            (
                "Choose the correct sentence.",
                ["I am learning English.", "I learning English."],
                0
            )
        ],

        "Tenses": [
            (
                "Which sentence is simple past?",
                ["I visited the site yesterday.", "I am visiting the site."],
                0
            ),
            (
                "Which sentence is present continuous?",
                ["I am studying English.", "I studied English."],
                0
            ),
            (
                "Which sentence is future?",
                ["I will study tomorrow.", "I studied yesterday."],
                0
            )
        ],

        "Vocabulary": [
            (
                "What does improve mean?",
                ["Make something better", "Stop something"],
                0
            ),
            (
                "Choose the natural sentence.",
                ["I feel confident.", "I feel confidence."],
                0
            ),
            (
                "Choose the natural sentence.",
                ["I want to develop my skills.",
                 "I want develop my skills."],
                0
            )
        ],

        "Sentence Formation": [
            (
                "Choose the natural sentence.",
                ["I practice English every day.",
                 "I every day practice English."],
                0
            ),
            (
                "Choose the natural sentence.",
                ["She works in Chennai.",
                 "She in Chennai works."],
                0
            ),
            (
                "Choose the natural sentence.",
                ["He is confident.",
                 "He confident is."],
                0
            )
        ]
    }

    topic = st.selectbox(
        "Choose topic",
        list(practice_data.keys())
    )

    questions = practice_data[topic]

    number = st.selectbox(
        "Choose question",
        range(1, len(questions) + 1),
        format_func=lambda x: f"Question {x}"
    )

    question, options, correct = questions[number - 1]

    st.markdown(
        f"### {question}"
    )

    answer = st.radio(
        "Choose your answer",
        options,
        key=f"practice_{topic}_{number}"
    )

    if st.button(
        "🔍 Check Answer",
        type="primary"
    ):
        is_correct = options.index(answer) == correct
        practice_score = 100 if is_correct else 0

        if is_correct:
            st.success("🎉 Good job! Your answer is correct.")
        else:
            st.info("Good attempt. Let's learn from this.")
            st.write(f"**Better answer:** {options[correct]}")

        save_activity(
            activity_type="Practice",
            activity=f"{topic} Practice - Q{number}",
            score=practice_score,
            review_data={
                "title": f"{topic} Practice",
                "question": question,
                "your_answer": answer,
                "correct_answer": options[correct],
                "result": "Correct" if is_correct else "Needs Review",
                "score": practice_score
            }
        )


# ============================================================
# STEP 10 - COMPLETE TEST & QUIZ COACH
# ============================================================

elif page == "🧪 Tests":

    friendly_coach_message("🧪 Tests")

    st.title("🧪 Tests & Quiz Coach")
    st.caption(
        "Choose a topic → answer 5 questions → get instant score, "
        "explanations and a recommended next step."
    )

    test_banks = {
        "Grammar": [
            ("Choose the correct sentence.",
             ["She go to work every day.", "She goes to work every day.",
              "She going to work every day.", "She gone to work every day."],
             "She goes to work every day.",
             "With 'she' in the simple present, use 'goes'."),
            ("Choose the correct past tense sentence.",
             ["I complete the work yesterday.", "I completed the work yesterday.",
              "I am completed the work yesterday.", "I completing the work yesterday."],
             "I completed the work yesterday.",
             "'Yesterday' requires the simple past here: completed."),
            ("Choose the correct article.",
             ["I am an engineer.", "I am a engineer.", "I am engineer.", "I an engineer."],
             "I am an engineer.",
             "Use 'an' before the vowel sound in 'engineer'."),
            ("Choose the correct future sentence.",
             ["I will attend the meeting tomorrow.", "I will attended the meeting tomorrow.",
              "I am will attend the meeting tomorrow.", "I will attending the meeting tomorrow."],
             "I will attend the meeting tomorrow.",
             "After 'will', use the base verb: attend."),
            ("Choose the correct present continuous sentence.",
             ["They are working now.", "They is working now.",
              "They working now.", "They are work now."],
             "They are working now.",
             "Present continuous uses are + verb-ing with 'they'."),
        ],
        "Tenses": [
            ("Which sentence is in the simple present tense?",
             ["I visit sites regularly.", "I visited the site.",
              "I am visiting the site.", "I will visit the site."],
             "I visit sites regularly.",
             "Simple present describes routines and repeated actions."),
            ("Which sentence is in the present perfect tense?",
             ["I completed the project.", "I have completed the project.",
              "I am completing the project.", "I will complete the project."],
             "I have completed the project.",
             "Present perfect uses have/has + past participle."),
            ("Choose the correct past continuous sentence.",
             ["I was working at 8 PM.", "I worked at 8 PM.",
              "I am working at 8 PM.", "I have worked at 8 PM."],
             "I was working at 8 PM.",
             "Past continuous uses was/were + verb-ing."),
            ("Choose the correct future perfect sentence.",
             ["I will finish the report.", "I will have finished the report by 6 PM.",
              "I have finished the report by 6 PM.", "I was finishing the report by 6 PM."],
             "I will have finished the report by 6 PM.",
             "Future perfect uses will have + past participle."),
            ("Which sentence describes an action continuing up to now?",
             ["I studied English.", "I have been studying English for two months.",
              "I will study English.", "I study English yesterday."],
             "I have been studying English for two months.",
             "Present perfect continuous fits an ongoing action with duration."),
        ],
        "Vocabulary": [
            ("What does 'reliable' mean?",
             ["Able to be trusted", "Very expensive", "Difficult to understand", "Quick to become angry"],
             "Able to be trusted", "Reliable means dependable or able to be trusted."),
            ("Choose the best word: 'I want to ___ my communication skills.'",
             ["improve", "damage", "avoid", "remove"],
             "improve", "Improve means make something better."),
            ("What is a professional synonym for 'help'?",
             ["assist", "hide", "delay", "refuse"],
             "assist", "Assist is a common professional synonym for help."),
            ("Choose the best word: 'We need a practical ___ to this problem.'",
             ["solution", "question", "delay", "argument"],
             "solution", "A solution is an answer or way to solve a problem."),
            ("What does 'collaborate' mean?",
             ["Work together", "Work alone", "Stop working", "Change jobs"],
             "Work together", "Collaborate means work jointly with others."),
        ],
        "Interview English": [
            ("Which is the strongest opening for 'Tell me about yourself'?",
             ["I don't know what to say.",
              "My name is Nishanth, and I have experience in the civil and cement industry.",
              "Everything is already in my resume.", "First I want to ask you something."],
             "My name is Nishanth, and I have experience in the civil and cement industry.",
             "A concise professional introduction gives the interviewer immediate context."),
            ("For a behavioral interview question, which structure is useful?",
             ["STAR", "ABC only", "Yes/No", "Random details"],
             "STAR", "STAR stands for Situation, Task, Action and Result."),
            ("Which answer style is generally clearer in an interview?",
             ["Direct answer → reason → example → result", "Long unrelated story",
              "One-word answer for every question", "Changing the topic"],
             "Direct answer → reason → example → result",
             "A structured answer is easier for the interviewer to follow."),
            ("Which sentence sounds more professional?",
             ["I don't know anything.",
              "I am still learning this area, and I am actively improving my knowledge.",
              "This question is not good.", "I never make mistakes."],
             "I am still learning this area, and I am actively improving my knowledge.",
             "It acknowledges a gap while showing a constructive learning attitude."),
            ("What should a strong example in an interview include?",
             ["A clear action and result", "Only background details",
              "Only technical words", "No outcome"],
             "A clear action and result",
             "Actions and results show what you actually did and the impact."),
        ],
    }

    topic = st.selectbox("Select test topic", list(test_banks.keys()), key="step10_test_topic")
    questions = test_banks[topic]

    st.subheader(f"📘 {topic} Test")
    st.write("Answer all 5 questions, then submit.")

    user_answers = []
    for index, (question, options, correct, explanation) in enumerate(questions, 1):
        selected = st.radio(
            f"Q{index}. {question}",
            ["Select an answer"] + options,
            key=f"step10_{topic}_{index}"
        )
        user_answers.append(selected)

    if st.button("✅ Submit Test", key="step10_submit_test", use_container_width=True):
        if "Select an answer" in user_answers:
            st.warning("Please answer all 5 questions before submitting.")
        else:
            results = []
            correct_count = 0

            for item, selected in zip(questions, user_answers):
                question, options, correct, explanation = item
                is_correct = selected == correct
                correct_count += int(is_correct)
                results.append({
                    "question": question,
                    "selected": selected,
                    "correct": correct,
                    "is_correct": is_correct,
                    "explanation": explanation,
                })

            score = round(correct_count / len(questions) * 100)
            st.session_state["step10_test_score"] = score
            st.session_state["step10_test_results"] = results
            st.session_state["step10_test_topic_result"] = topic

            save_activity(
                activity_type="Test",
                activity=f"{topic} Test",
                score=score,
                review_data={
                    "title": f"{topic} Test",
                    "topic": topic,
                    "score": score,
                    "correct_count": correct_count,
                    "total_questions": len(questions),
                    "questions": results
                }
            )
            st.rerun()

    saved_score = st.session_state.get("step10_test_score")
    saved_results = st.session_state.get("step10_test_results", [])
    saved_topic = st.session_state.get("step10_test_topic_result")

    if saved_score is not None and saved_results:
        st.divider()
        st.subheader("📊 Test Result")
        st.metric(f"{saved_topic} Score", f"{saved_score}%")

        correct_total = sum(1 for result in saved_results if result["is_correct"])
        st.write(f"Correct answers: **{correct_total}/5**")

        if saved_score >= 80:
            st.success("Excellent. Move to another topic or a harder practice.")
        elif saved_score >= 60:
            st.info("Good progress. Review the incorrect answers once.")
        else:
            st.warning("Review the explanations and practice the topic before retrying.")

        st.subheader("🧠 Answer Review")
        for number, result in enumerate(saved_results, 1):
            with st.container(border=True):
                status = "✅ Correct" if result["is_correct"] else "❌ Incorrect"
                st.markdown(f"**Q{number}. {status}**")
                st.write(result["question"])
                st.write(f"Your answer: **{result['selected']}**")
                if not result["is_correct"]:
                    st.write(f"Correct answer: **{result['correct']}**")
                st.caption(f"Why: {result['explanation']}")

        st.subheader("🎯 Recommended Next Step")
        if all(result["is_correct"] for result in saved_results):
            st.write(
                f"You answered all {saved_topic} questions correctly. "
                "Choose another topic for your next test."
            )
        else:
            st.write(
                f"Review the **{saved_topic}** explanations above, practice the "
                "incorrect concepts, and attempt the test again later."
            )

    st.divider()
    st.subheader("📈 Test Progress")

    db = SessionLocal()
    try:
        test_history = (
            db.query(ActivityLog)
            .filter(
                ActivityLog.user_id == st.session_state.user_id,
                ActivityLog.activity_type == "Test",
                ActivityLog.score.isnot(None)
            )
            .order_by(ActivityLog.created_at.desc())
            .all()
        )
    finally:
        db.close()

    if test_history:
        scores = [log.score for log in test_history]
        t1, t2, t3 = st.columns(3)
        t1.metric("Tests Completed", len(test_history))
        t2.metric("Average", f"{round(sum(scores) / len(scores))}%")
        t3.metric("Best", f"{max(scores)}%")
        st.caption(
            "Test results are saved to the same learning history used by "
            "Dashboard, Progress and History."
        )
    else:
        st.info("Complete a test to start tracking your test progress.")


# ============================================================
# STEP 9 - COMPLETE DAILY ASSIGNMENT COACH
# ============================================================

elif page == "📝 Daily Assignment":

    friendly_coach_message("📝 Daily Assignment")

    st.title("📝 Daily Assignment")
    st.caption(
        "One focused assignment each day → AI feedback → score → improvement."
    )

    db = SessionLocal()
    try:
        assignment_logs = (
            db.query(ActivityLog)
            .filter(
                ActivityLog.user_id == st.session_state.user_id,
                ActivityLog.activity_type == "Assignment"
            )
            .order_by(ActivityLog.created_at.desc())
            .all()
        )

        scored_logs = (
            db.query(ActivityLog)
            .filter(
                ActivityLog.user_id == st.session_state.user_id,
                ActivityLog.score.isnot(None)
            )
            .all()
        )
    finally:
        db.close()

    # Detect current focus from scored learning data.
    focus_groups = {
        "Speaking": [
            log.score for log in scored_logs
            if log.activity_type == "Speaking"
        ],
        "Interview": [
            log.score for log in scored_logs
            if log.activity_type == "Interview"
        ],
        "Tests": [
            log.score for log in scored_logs
            if log.activity_type == "Test"
        ],
    }

    focus_avgs = {
        name: round(sum(values) / len(values))
        for name, values in focus_groups.items()
        if values
    }

    if focus_avgs:
        daily_focus = min(focus_avgs, key=focus_avgs.get)
        daily_focus_score = focus_avgs[daily_focus]
    else:
        daily_focus = "Speaking"
        daily_focus_score = None

    # IST day for duplicate-safe daily assignment.
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    today_ist = now_ist.date()

    completed_today = False
    today_assignment_log = None

    for log in assignment_logs:
        if log.created_at is None:
            continue
        log_ist = log.created_at + timedelta(hours=5, minutes=30)
        if log_ist.date() == today_ist:
            completed_today = True
            today_assignment_log = log
            break

    st.subheader("🎯 Today's Focus")

    if daily_focus_score is not None:
        st.info(
            f"Current priority: **{daily_focus} ({daily_focus_score}%)**. "
            "Today's assignment is designed around this area."
        )
    else:
        st.info(
            "Today's assignment starts with Speaking. Your future assignments "
            "will adapt to your saved scores."
        )

    if daily_focus == "Interview":
        assignment_title = "Structured Interview Answer"
        assignment_prompt = (
            "Write an answer to: **Why should we hire you?**\n\n"
            "Use this structure:\n"
            "1. Direct answer\n"
            "2. Your relevant strength\n"
            "3. One short example\n"
            "4. How you can add value\n\n"
            "Target: 80–150 words."
        )
        assignment_system = (
            "You are an English interview coach. Evaluate the user's written "
            "interview answer. Score only the written text. Do not evaluate "
            "pronunciation, accent, voice, eye contact or body language."
        )
        breakdown = (
            "- Relevance: X/20\n"
            "- Structure: X/20\n"
            "- Grammar: X/20\n"
            "- Vocabulary: X/20\n"
            "- Interview Impact: X/20"
        )
    elif daily_focus == "Speaking":
        assignment_title = "Natural Speaking Answer"
        assignment_prompt = (
            "Write how you would naturally answer: "
            "**Tell me about your daily routine.**\n\n"
            "Use simple spoken English with a clear beginning, middle and ending.\n"
            "Target: 80–150 words."
        )
        assignment_system = (
            "You are an English speaking coach. Evaluate the user's written "
            "practice as a speaking script. Score only the text. Do not evaluate "
            "pronunciation, accent, voice or speaking speed."
        )
        breakdown = (
            "- Grammar: X/20\n"
            "- Sentence Formation: X/20\n"
            "- Vocabulary: X/20\n"
            "- Clarity: X/20\n"
            "- Topic Relevance: X/20"
        )
    else:
        assignment_title = "Grammar in Real Sentences"
        assignment_prompt = (
            "Write **8 original sentences** about work, learning or daily life.\n\n"
            "Use a mix of present, past and future tense. "
            "Keep the sentences natural and practical."
        )
        assignment_system = (
            "You are an English grammar coach. Evaluate the user's written "
            "sentences for correct grammar, tense usage, clarity and naturalness."
        )
        breakdown = (
            "- Grammar Accuracy: X/20\n"
            "- Tense Usage: X/20\n"
            "- Sentence Formation: X/20\n"
            "- Vocabulary: X/20\n"
            "- Clarity: X/20"
        )

    with st.container(border=True):
        st.markdown(f"### 📌 {assignment_title}")
        st.markdown(assignment_prompt)

    if completed_today:
        score_text = (
            f"{int(today_assignment_log.score)}%"
            if today_assignment_log.score is not None
            else "Completed"
        )
        st.success(
            f"✅ Today's assignment is already saved. Result: **{score_text}**"
        )
        st.caption(
            "A new daily assignment will be available on the next IST calendar day."
        )
    else:
        assignment_reset = "step14_assignment_reset"
        assignment_key = f"step9_daily_answer_{st.session_state.get(assignment_reset, 0)}"
        answer = st.text_area("✍️ Your answer", height=220, key=assignment_key, placeholder="Write your answer here...")
        clear_col, hint_col = st.columns(2)
        with clear_col:
            if st.button("🗑️ Clear & Start Again", key=f"clear_{assignment_key}", use_container_width=True):
                reset_counter(assignment_reset)
        with hint_col:
            if st.button("💡 Help Me Answer in English", key="step14_assignment_hint", use_container_width=True):
                st.info("Start with one direct sentence. Then add one reason, one short example, and the value or result.")

        word_count = len(answer.split()) if answer.strip() else 0
        st.caption(f"Word count: {word_count}")

        if st.button(
            "🤖 Submit to AI Coach",
            key="step9_submit_assignment",
            use_container_width=True
        ):
            if not answer.strip():
                st.info("🌱 Write a short answer first. Start simple — your coach will help you improve it.")
            elif not show_english_retry_support(answer):
                pass
            elif word_count < 20:
                st.warning(
                    "Good start. Add a little more detail — around 20 words helps the "
                    "coach can give useful feedback."
                )
            else:
                evaluation_prompt = f"""
Evaluate this daily English assignment.

Assignment:
{assignment_title}

User answer:
{answer}

Return EXACTLY this format:

OVERALL_SCORE: <integer 0-100>

## Score Breakdown
{breakdown}

## What You Did Well
- Give 2 or 3 specific points.

## Corrections
- Show the important grammar, sentence or wording corrections.

## Better Version
Rewrite the answer in natural, simple English while keeping the user's meaning.

## Today's Coach Tip
Give 2 short practical tips for the next practice.
"""

                with st.spinner("AI Coach is checking your assignment..."):
                    report = ask_ai(
                        assignment_system,
                        evaluation_prompt
                    )

                if report:
                    match = re.search(
                        r"OVERALL_SCORE:\s*(\d{1,3})",
                        report,
                        flags=re.IGNORECASE
                    )

                    if not match:
                        retry_prompt = (
                            evaluation_prompt
                            + "\nIMPORTANT: The first line MUST be "
                            + "OVERALL_SCORE: <integer 0-100>."
                        )
                        report = ask_ai(
                            assignment_system,
                            retry_prompt
                        )
                        match = re.search(
                            r"OVERALL_SCORE:\s*(\d{1,3})",
                            report or "",
                            flags=re.IGNORECASE
                        )

                    if match:

                        # ====================================
                        # DAILY ASSIGNMENT SCORE SCALE
                        # GUARDRAIL V1
                        # ====================================
                        #
                        # Existing Daily Assignment AI still
                        # creates the score and feedback.
                        #
                        # This layer validates the score before
                        # saving it to session state / database.
                        #
                        # It never blindly converts 8 -> 80.
                        # ====================================

                        _raw_assignment_score = match.group(1)

                        _assignment_score_guard = (
                            validate_score_contract(
                                _raw_assignment_score
                            )
                        )

                        if not _assignment_score_guard.get(
                            "valid"
                        ):

                            _assignment_guard_status = (
                                _assignment_score_guard.get(
                                    "status",
                                    "UNKNOWN"
                                )
                            )

                            st.error(
                                "The AI returned an ambiguous "
                                "or invalid assignment score. "
                                f"Guardrail status: "
                                f"{_assignment_guard_status}. "
                                "Please submit again."
                            )

                        else:

                            assignment_score = int(
                                round(
                                    float(
                                        _assignment_score_guard[
                                            "score"
                                        ]
                                    )
                                )
                            )

                            clean_report = re.sub(
                                r"^\s*OVERALL_SCORE:\s*\d{1,3}\s*",
                                "",
                                report,
                                count=1,
                                flags=re.IGNORECASE
                            ).strip()

                            st.session_state["step9_assignment_report"] = clean_report
                            st.session_state["step9_assignment_score"] = assignment_score

                            save_activity(
                                activity_type="Assignment",
                                activity=f"Daily Assignment - {assignment_title}",
                                score=assignment_score,
                                review_data={
                                    "title": assignment_title,
                                    "question": assignment_prompt,
                                    "your_answer": answer,
                                    "ai_feedback": clean_report,
                                    "score": assignment_score
                                }
                            )

                            st.rerun()
                    else:
                        st.error(
                            "The AI response did not return a valid score. "
                            "Please submit again."
                        )
                else:
                    st.error(
                        "The AI Coach did not return a response. Please try again."
                    )

    saved_report = st.session_state.get("step9_assignment_report", "")
    saved_score = st.session_state.get("step9_assignment_score")

    if saved_report and saved_score is not None:
        st.divider()
        st.subheader("📊 Today's Assignment Result")
        st.metric("AI Assignment Score", f"{saved_score}%")
        st.markdown(saved_report)

    st.divider()
    st.subheader("📚 Assignment Progress")

    scored_assignments = [
        log.score for log in assignment_logs
        if log.score is not None
    ]

    if scored_assignments:
        a1, a2, a3 = st.columns(3)
        a1.metric("Completed", len(assignment_logs))
        a2.metric(
            "Average",
            f"{round(sum(scored_assignments) / len(scored_assignments))}%"
        )
        a3.metric("Best", f"{max(scored_assignments)}%")
    else:
        st.info(
            "Complete your first AI-scored daily assignment to start tracking progress."
        )

# ============================================================
# SPEAKING
# ============================================================

elif page == "🗣️ Speaking":

    friendly_coach_message("🗣️ Speaking")

    st.subheader("🎙️ Microphone Practice")
    st.caption(
        "Step 3A: Record your voice and play it back. "
        "Speech-to-text and pronunciation analysis will be added next."
    )

    expected_voice_text = st.text_input(
        "Practice sentence to read aloud",
        value="I want to improve my English every day.",
        key="voice_accuracy_expected_text"
    )
    st.caption("Read the sentence above exactly, then record your voice.")

    if hasattr(st, "audio_input"):
        speaking_audio = st.audio_input(
            "Record your speaking practice",
            key="speaking_audio_input"
        )

        if speaking_audio is not None:
            st.success("✅ Audio recorded successfully.")
            st.audio(speaking_audio)
            audio_bytes = speaking_audio.getvalue()
            st.caption(f"Recording captured: {len(audio_bytes) / 1024:.1f} KB")

            st.subheader("📝 Speech-to-Text")

            if sr is None:
                st.warning(
                    "SpeechRecognition is not installed yet. "
                    "Run: pip install SpeechRecognition"
                )
            else:
                if st.button(
                    "🗣️ Convert My Audio to Text",
                    key="convert_speaking_audio"
                ):
                    try:
                        recognizer = sr.Recognizer()

                        # st.audio_input provides WAV audio, which can be read
                        # directly by SpeechRecognition.
                        speaking_audio.seek(0)

                        with sr.AudioFile(speaking_audio) as source:
                            audio_data = recognizer.record(source)

                        with st.spinner("Converting your voice to text..."):
                            transcript = recognizer.recognize_google(
                                audio_data,
                                language="en-IN"
                            )

                        if voice_transcript_is_english(transcript):
                            st.session_state["speaking_transcript"] = transcript
                            st.success("✅ English speech captured successfully.")
                        else:
                            st.session_state.pop("speaking_transcript", None)
                            show_english_voice_message()

                    except sr.UnknownValueError:
                        st.info(
                            "🤝 I couldn’t catch that clearly. Try again in simple English, a little slower and closer to the microphone."
                        )
                    except sr.RequestError as e:
                        st.info(
                            "🎙️ Speech recognition is temporarily unavailable. "
                            "Please check your internet connection and try again."
                        )
                    except Exception as e:
                        st.info("🎙️ Audio could not be processed this time. Please record again and retry.")

                transcript = st.session_state.get(
                    "speaking_transcript", ""
                )

                if transcript:
                    st.text_area(
                        "Your Transcript",
                        value=transcript,
                        height=120,
                        key="speaking_transcript_view"
                    )
                    st.caption(
                        "Step 3B only converts audio to text. "
                        "Connecting this transcript to AI Speaking scoring "
                        "will be Step 3C."
                    )

                    st.markdown("### 🎯 Voice Accuracy & Pronunciation Practice")
                    st.caption(
                        "Voice Accuracy compares your practice sentence with the speech-to-text "
                        "transcript. It is not a phoneme-level pronunciation score."
                    )

                    accuracy = calculate_voice_accuracy(expected_voice_text, transcript)
                    st.write("**Expected:**", expected_voice_text)
                    st.write("**Heard:**", transcript)
                    st.metric("Voice Accuracy", f"{accuracy['score']}%")
                    st.write(f"**Correct words:** {accuracy['correct']}/{accuracy['total']}")

                    if accuracy["missing"]:
                        st.write("**Missing words:**", ", ".join(accuracy["missing"]))
                    else:
                        st.write("**Missing words:** None")

                    if accuracy["different"]:
                        st.write("**Different / extra words:**")
                        for expected_part, heard_part in accuracy["different"]:
                            if expected_part == "(extra)":
                                st.write(f"- Extra heard: {heard_part}")
                            else:
                                st.write(
                                    f"- Expected: {expected_part} → "
                                    f"Heard: {heard_part or '(not detected)'}"
                                )
                    else:
                        st.write("**Different / extra words:** None")

                    if accuracy["score"] >= 90:
                        st.success("Very clear match. Keep the same natural pace and clarity.")
                    elif accuracy["score"] >= 70:
                        st.info("Good match. Practice the missing or different words once more.")
                    else:
                        st.warning(
                            "The transcript differs from the expected sentence. "
                            "Speak a little more clearly and retry the difficult words."
                        )

                    st.markdown(
                        "**Coach Tips:** Speak naturally, avoid rushing, and repeat the "
                        "words that were missed or heard differently."
                    )

                    if st.button(
                        "🤖 Analyze My Voice Transcript",
                        key="analyze_voice_transcript"
                    ):
                        if not show_english_retry_support(transcript):
                            st.session_state.pop("voice_ai_score", None)
                            st.session_state.pop("voice_ai_report", None)
                            st.stop()
                        try:
                            voice_system = """
You are an English Speaking Coach.

Evaluate ONLY the learner's transcribed spoken answer below.
This is text-based language-content scoring from a voice transcript.
Do NOT score pronunciation, accent, voice quality, speaking speed,
pauses, eye contact, or body language.

Score these five areas from 0 to 20:
1. Grammar
2. Sentence Formation
3. Vocabulary
4. Clarity
5. Topic Relevance

The five scores must add up to an overall score from 0 to 100.

Return ONLY clean Markdown.
The FIRST line MUST be exactly:
OVERALL_SCORE: <integer from 0 to 100>

Then use:

## Score Breakdown
- Grammar: <score>/20
- Sentence Formation: <score>/20
- Vocabulary: <score>/20
- Clarity: <score>/20
- Relevance: <score>/20

## Communication
Brief feedback on clarity and naturalness.

## What You Did Well
Give specific strengths.

## English Corrections
Use:
- Original: ...
- Better: ...
- Why: ...

## Natural Speaking Version
Give a natural corrected version without inventing personal details.

## Coach Tips
Give 3 short actionable tips.

## Next Practice
Give one short practice task.
"""

                            voice_topic = st.session_state.get(
                                "current_speaking_topic",
                                "Voice Speaking Practice"
                            )

                            voice_user = f"""
Speaking topic:
{voice_topic}

Voice transcript:
{transcript}
"""

                            voice_report = ask_ai(
                                voice_system,
                                voice_user
                            )

                            if (
                                voice_report is None
                                or not isinstance(voice_report, str)
                                or not voice_report.strip()
                            ):
                                retry_system = """
Evaluate the learner's English voice transcript as TEXT only.
Do not score pronunciation or audio quality.

Score Grammar, Sentence Formation, Vocabulary,
Clarity and Relevance from 0 to 20 each.

First line exactly:
OVERALL_SCORE: <0-100>

Then give:
## Score Breakdown
## Communication
## What You Did Well
## English Corrections
## Natural Speaking Version
## Coach Tips
## Next Practice
"""
                                voice_report = ask_ai(
                                    retry_system,
                                    voice_user
                                )

                            if (
                                voice_report is None
                                or not isinstance(voice_report, str)
                                or not voice_report.strip()
                            ):
                                raise ValueError(
                                    "AI service returned an empty response. "
                                    "Please try again."
                                )

                            voice_score_match = re.search(
                                r"OVERALL_SCORE:\s*(\d{1,3})",
                                voice_report,
                                flags=re.IGNORECASE
                            )

                            if not voice_score_match:
                                raise ValueError(
                                    "AI response was received, but the score "
                                    "format was missing. Please try again."
                                )

                            voice_score = int(
                                voice_score_match.group(1)
                            )
                            voice_score = max(
                                0, min(100, voice_score)
                            )

                            clean_voice_report = re.sub(
                                r"^\s*OVERALL_SCORE:\s*\d{1,3}\s*",
                                "",
                                voice_report,
                                count=1,
                                flags=re.IGNORECASE
                            ).strip()

                            st.session_state[
                                "voice_ai_score"
                            ] = voice_score
                            st.session_state[
                                "voice_ai_report"
                            ] = clean_voice_report

                            save_activity(
                                activity_type="Speaking",
                                activity=f"{voice_topic} - Voice",
                                score=voice_score,
                                review_data={
                                    "title": f"{voice_topic} - Voice Practice",
                                    "topic": voice_topic,
                                    "transcript": transcript,
                                    "ai_feedback": clean_voice_report,
                                    "score": voice_score
                                }
                            )

                        except Exception as e:
                            st.error(
                                f"Voice AI analysis failed: {e}"
                            )

                    voice_ai_score = st.session_state.get(
                        "voice_ai_score"
                    )
                    voice_ai_report = st.session_state.get(
                        "voice_ai_report", ""
                    )

                    if transcript:
                        st.divider()
                        st.subheader("🎧 Fluency Coach")

                        # Estimate recording duration from the WAV header.
                        # Streamlit audio_input returns WAV audio.
                        audio_duration = None
                        try:
                            import io
                            import wave

                            with wave.open(
                                io.BytesIO(audio_bytes), "rb"
                            ) as wav_file:
                                frames = wav_file.getnframes()
                                rate = wav_file.getframerate()
                                if rate:
                                    audio_duration = frames / float(rate)
                        except Exception:
                            audio_duration = None

                        words = [
                            w for w in transcript.strip().split()
                            if w.strip()
                        ]
                        word_count = len(words)

                        filler_phrases = [
                            "um", "uh", "erm", "hmm",
                            "you know", "i mean", "basically",
                            "actually", "like"
                        ]

                        lower_transcript = transcript.lower()
                        filler_count = 0

                        # Count multi-word fillers first.
                        for phrase in ["you know", "i mean"]:
                            filler_count += lower_transcript.count(phrase)
                            lower_transcript = lower_transcript.replace(
                                phrase, " "
                            )

                        # Count single-word fillers conservatively.
                        clean_words = [
                            w.strip(".,!?;:'\"()[]{}").lower()
                            for w in lower_transcript.split()
                        ]
                        for filler in [
                            "um", "uh", "erm", "hmm",
                            "basically", "actually", "like"
                        ]:
                            filler_count += clean_words.count(filler)

                        wpm = None
                        if (
                            audio_duration is not None
                            and audio_duration > 0
                            and word_count > 0
                        ):
                            wpm = round(
                                word_count / (audio_duration / 60.0)
                            )

                        # Fluency score is based only on measurable pace,
                        # continuity proxy and filler usage.
                        pace_score = 20
                        if wpm is not None:
                            if 90 <= wpm <= 160:
                                pace_score = 20
                            elif 70 <= wpm < 90 or 160 < wpm <= 180:
                                pace_score = 16
                            elif 50 <= wpm < 70 or 180 < wpm <= 200:
                                pace_score = 12
                            else:
                                pace_score = 8

                        filler_score = 20
                        if word_count > 0:
                            filler_ratio = filler_count / word_count
                            if filler_ratio == 0:
                                filler_score = 20
                            elif filler_ratio <= 0.03:
                                filler_score = 17
                            elif filler_ratio <= 0.06:
                                filler_score = 13
                            else:
                                filler_score = 9

                        length_score = 20
                        if word_count < 5:
                            length_score = 8
                        elif word_count < 10:
                            length_score = 12
                        elif word_count < 20:
                            length_score = 16

                        # Keep this explicitly a fluency indicator, not a
                        # pronunciation score.
                        fluency_score = round(
                            (
                                pace_score
                                + filler_score
                                + length_score
                            ) / 60 * 100
                        )

                        col_f1, col_f2, col_f3 = st.columns(3)

                        with col_f1:
                            st.metric(
                                "⏱️ Duration",
                                (
                                    f"{audio_duration:.1f} sec"
                                    if audio_duration is not None
                                    else "Unavailable"
                                )
                            )

                        with col_f2:
                            st.metric(
                                "🗣️ Speaking Pace",
                                (
                                    f"{wpm} WPM"
                                    if wpm is not None
                                    else "Unavailable"
                                )
                            )

                        with col_f3:
                            st.metric(
                                "💬 Fillers Detected",
                                filler_count
                            )

                        st.metric(
                            "🎯 Fluency Indicator",
                            f"{fluency_score}%"
                        )

                        if wpm is not None:
                            if wpm < 90:
                                st.info(
                                    "💡 Pace Tip: Your pace appears slow. "
                                    "Practice speaking in short connected "
                                    "sentences without long gaps."
                                )
                            elif wpm > 160:
                                st.info(
                                    "💡 Pace Tip: Your pace appears fast. "
                                    "Slow down slightly and give each idea "
                                    "enough space."
                                )
                            else:
                                st.success(
                                    "✅ Your speaking pace is within the "
                                    "practice target range."
                                )

                        if filler_count > 0:
                            st.info(
                                "💡 Filler Tip: Try a short silent pause "
                                "instead of filler words when you need time "
                                "to think."
                            )
                        else:
                            st.success(
                                "✅ No common filler words were detected "
                                "in the transcript."
                            )

                        st.caption(
                            "Fluency Indicator uses measurable recording "
                            "duration, speaking pace, answer length and "
                            "transcribed filler words. It is not a "
                            "pronunciation-accuracy score."
                        )

                        st.info(
                            "🔤 Pronunciation Coach: This version does not "
                            "claim word-by-word pronunciation accuracy. "
                            "Reliable pronunciation scoring needs phoneme- "
                            "or word-level speech assessment, which will be "
                            "added separately."
                        )

                    if (
                        voice_ai_score is not None
                        and voice_ai_report
                    ):
                        st.divider()
                        st.subheader("📊 Voice Transcript Feedback")
                        st.metric(
                            "🎯 Text-Based Voice Speaking Score",
                            f"{voice_ai_score}%"
                        )
                        st.caption(
                            "This score evaluates the English content from "
                            "your voice transcript. Pronunciation and real "
                            "audio fluency scoring will be added in Step 3D."
                        )
                        st.markdown(voice_ai_report)

    else:
        st.warning(
            "Native microphone recording is not available in this "
            "Streamlit installation."
        )

    st.divider()

    st.title("🗣️ Speaking Practice")
    st.caption("Practice → Analyze → Improve → Retry")

    mode = st.radio(
        "Choose mode",
        ["🎤 Guided Speaking", "💬 Open Speaking"],
        horizontal=True
    )

    if mode == "🎤 Guided Speaking":
        st.markdown("### 🎯 Step 4A — Dynamic Speaking Questions")

        speaking_level = st.selectbox(
            "Choose your speaking level",
            ["Beginner", "Intermediate", "Advanced"],
            key="dynamic_speaking_level"
        )

        question_bank = {
            "Beginner": [
                "What do you usually do in the morning?",
                "Tell me about your family.",
                "What is your favorite food and why?",
                "How do you spend your free time?",
                "Describe your hometown.",
                "What do you do at work or during your studies?",
                "What is one thing you want to improve in English?",
                "Tell me about a good day you remember.",
                "What are your plans for this weekend?",
                "Describe a person who helps you."
            ],
            "Intermediate": [
                "Describe a challenge you faced and how you handled it.",
                "What skill are you currently learning, and why is it important to you?",
                "Explain one project or task you are proud of.",
                "How has technology changed the way people work?",
                "What are your career goals for the next few years?",
                "Describe a problem you solved at work or in daily life.",
                "What makes a good team member?",
                "How do you manage your time when you have many tasks?",
                "What are the advantages and disadvantages of working from home?",
                "Describe something new you learned recently."
            ],
            "Advanced": [
                "How can artificial intelligence improve traditional industries?",
                "Explain a difficult decision you made and the factors you considered.",
                "What are the most important qualities of an effective leader?",
                "How would you improve a process that is slow or inefficient?",
                "Discuss the benefits and risks of using AI in the workplace.",
                "How should companies balance technology, cost, and customer experience?",
                "Describe a situation where data could help make a better business decision.",
                "What changes do you expect in your industry over the next five years?",
                "How would you convince a team to adopt a new technology?",
                "Explain how continuous learning can affect long-term career growth."
            ]
        }

        level_key = f"speaking_question_index_{speaking_level}"
        if level_key not in st.session_state:
            st.session_state[level_key] = 0

        questions = question_bank[speaking_level]
        question_index = st.session_state[level_key] % len(questions)
        topic = questions[question_index]

        st.info(f"🎤 Speaking Question: {topic}")

        if st.button(
            "🔄 Next Question",
            key=f"next_speaking_question_{speaking_level}"
        ):
            st.session_state[level_key] = (
                st.session_state[level_key] + 1
            ) % len(questions)
            st.session_state.pop("speaking_transcript", None)
            st.session_state.pop("voice_ai_score", None)
            st.session_state.pop("voice_ai_report", None)
            st.rerun()

        st.caption(
            "Answer naturally in your own words. "
            "The existing AI coach will evaluate your English content."
        )

    else:
        topic = "Open Conversation"
        st.info("💬 Speak freely about any topic.")

    # Keep the selected topic available to the microphone/voice AI block
    # across Streamlit reruns.
    # STEP 4C - while retrying, keep the original question fixed.
    if st.session_state.get("step4_retry_mode", False):
        retry_question = st.session_state.get("step4_baseline_question")
        if retry_question:
            topic = retry_question
            st.warning(f"🔁 Retry Question: {topic}")

    # STEP 4B - use the latest AI-scored voice answer to choose a focus area.
    latest_speaking_report = st.session_state.get("speaking_report", "")
    latest_voice_report = st.session_state.get("voice_ai_report", "")
    adaptive_source_report = latest_speaking_report or latest_voice_report
    adaptive_focus, adaptive_focus_score = get_adaptive_focus(adaptive_source_report)

    if mode == "🎤 Guided Speaking":
        st.markdown("### 🧠 Adaptive Speaking Coach")

        if adaptive_focus:
            st.write(f"**Current Focus Area:** {adaptive_focus}")
            st.write(f"**Latest Area Score:** {adaptive_focus_score}/20")
            st.info(adaptive_coach_tip(adaptive_focus))
            st.caption(
                "🎯 Focus especially on this area in your next answer. "
                "The AI coach will still evaluate all speaking areas."
            )

            if not st.session_state.get("step4_retry_mode", False):
                if st.button("🎯 Retry With Coach Tip", key="step4_retry_button"):
                    st.session_state["step4_retry_mode"] = True
                    st.session_state["step4_baseline_score"] = st.session_state.get("speaking_score")
                    st.session_state["step4_baseline_focus"] = adaptive_focus
                    st.session_state["step4_baseline_question"] = topic
                    st.session_state["step4_retry_result_score"] = None
                    st.session_state["speaking_report"] = ""
                    st.session_state["speaking_score"] = None
                    st.rerun()
            else:
                st.success("🔁 Retry Mode: Answer the same question again using the coach tip.")
                baseline = st.session_state.get("step4_baseline_score")
                if baseline is not None:
                    st.caption(f"Previous score: {baseline}% — this retry will be compared automatically.")
        else:
            st.info(
                "Complete one AI-scored speaking answer first. "
                "Then the coach will automatically identify your next focus area."
            )

    # STEP 4D/4E - compare the completed retry with its baseline.
    baseline_score = st.session_state.get("step4_baseline_score")
    retry_score = st.session_state.get("step4_retry_result_score")

    if baseline_score is not None and retry_score is not None:
        st.markdown("### 📈 Retry Improvement")
        difference = retry_score - baseline_score

        c1, c2, c3 = st.columns(3)
        c1.metric("Previous", f"{baseline_score}%")
        c2.metric("Retry", f"{retry_score}%")
        c3.metric("Change", f"{difference:+d}%")

        if difference > 0:
            st.success(
                f"✅ Improved by {difference}%. Continue using the coach structure, "
                "then move to the next question."
            )
        elif difference == 0:
            st.info(
                "Your score stayed the same. Try the same question once more with "
                "shorter, clearer sentences."
            )
        else:
            st.warning(
                "This attempt gives us another learning point. Focus only on the coach tip "
                "and use simple sentences before adding more detail."
            )

        if st.button("➡️ Finish Retry & Continue", key="step4_finish_retry"):
            st.session_state["step4_baseline_score"] = None
            st.session_state["step4_baseline_focus"] = None
            st.session_state["step4_baseline_question"] = None
            st.session_state["step4_retry_result_score"] = None
            st.session_state["speaking_report"] = ""
            st.session_state["speaking_score"] = None
            st.rerun()

    st.session_state["current_speaking_topic"] = topic

    speaking_text_reset = "step14_speaking_text_reset"
    speaking_text_key = f"step14_speaking_answer_{st.session_state.get(speaking_text_reset, 0)}"
    transcript = st.text_area("📝 Your Answer", height=220, key=speaking_text_key,
        placeholder="Type your answer here. Start with your main point, then add a reason and one example.")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("🗑️ Clear & Try Again", key=f"clear_{speaking_text_key}", use_container_width=True):
            reset_counter(speaking_text_reset)
    with sc2:
        if st.button("💡 Help Me Answer in English", key="step14_speaking_hint", use_container_width=True):
            st.info("Think of your idea first, then use simple English: main point → reason → one example.")

    st.caption(
        "💡 Type your answer here for text practice, or use the microphone "
        "practice above for voice-based coaching."
    )

    if st.button(
        "🤖 Analyze My English",
        type="primary",
        use_container_width=True
    ):
        if not transcript.strip():
            st.info("🤝 Add a short answer first — even 2–3 simple sentences are enough to begin.")
        elif not show_english_retry_support(transcript):
            pass
        else:
            speaking_system = """
You are a friendly English Speaking Coach.

Analyze only the learner's typed speaking answer.
Do not invent personal details.
This is TEXT-BASED language scoring only.
Do NOT score pronunciation, accent, voice quality, speaking speed,
eye contact, body language, microphone quality, or camera performance.

Evaluate these five areas from 0 to 20:
1. Grammar
2. Sentence Formation
3. Vocabulary
4. Clarity
5. Relevance to Topic

The five scores must add up to an Overall Score from 0 to 100.

Return ONLY clean Markdown.
The FIRST line MUST be exactly:
OVERALL_SCORE: <integer from 0 to 100>

Then use exactly these sections:

## Score Breakdown
- Grammar: <score>/20
- Sentence Formation: <score>/20
- Vocabulary: <score>/20
- Clarity: <score>/20
- Relevance: <score>/20

## Communication
Briefly describe clarity and flow based only on the written answer.

## What You Did Well
Give 3 specific positive points.

## English Corrections
Show important corrections in this format:
- Original: ...
- Better: ...
- Why: ...

Do not correct every sentence. Focus on the most useful mistakes.

## Natural Speaking Version
Rewrite the answer in simple, natural spoken English while keeping the
learner's original meaning and facts.

## Coach Tips
Give 3 short practical speaking tips.

## Next Practice
Give one short speaking task.

Be supportive, practical and easy for a beginner to understand.
"""
            speaking_user = f"""
Topic:
{topic}

Learner's answer:
{transcript}

Analyze this answer now.
"""

            with st.spinner("🤖 Analyzing your English..."):
                try:
                    report = ask_ai(speaking_system, speaking_user)

                    # OpenRouter/AI engine may occasionally return None or an
                    # empty response. Never pass that into regex.
                    if report is None or not isinstance(report, str) or not report.strip():
                        retry_system = """You are an English speaking coach.
Evaluate the learner's WRITTEN answer only.
Do not score pronunciation, accent, audio, eye contact or body language.
Score Grammar, Sentence Formation, Vocabulary, Clarity and Relevance
from 0 to 20 each.

Return the first line exactly as:
OVERALL_SCORE: <0-100>

Then give:
## Score Breakdown
- Grammar: <0-20>/20
- Sentence Formation: <0-20>/20
- Vocabulary: <0-20>/20
- Clarity: <0-20>/20
- Relevance: <0-20>/20

Then briefly give:
## Communication
## What You Did Well
## English Corrections
## Natural Speaking Version
## Coach Tips
## Next Practice
"""
                        report = ask_ai(retry_system, speaking_user)

                    if report is None or not isinstance(report, str) or not report.strip():
                        raise ValueError(
                            "AI service returned an empty response. "
                            "Please try Analyze My English again."
                        )

                    score_match = re.search(
                        r"OVERALL_SCORE:\s*(\d{1,3})",
                        report,
                        flags=re.IGNORECASE
                    )

                    if not score_match:
                        raise ValueError(
                            "AI response was received, but the score format was missing. "
                            "Please try Analyze My English again."
                        )

                    # ====================================
                    # SPEAKING SCORE SCALE GUARDRAIL V1
                    # ====================================
                    #
                    # Existing Speaking AI still generates
                    # the score and full coaching report.
                    #
                    # This layer validates the score before
                    # it is saved to session state / database.
                    #
                    # No blind 8 -> 80 conversion.
                    # No camera / audio scoring is added.
                    # ====================================

                    _raw_speaking_score = score_match.group(1)

                    _speaking_score_guard = validate_score_contract(
                        _raw_speaking_score
                    )

                    if not _speaking_score_guard.get("valid"):

                        _speaking_guard_status = (
                            _speaking_score_guard.get(
                                "status",
                                "UNKNOWN"
                            )
                        )

                        raise ValueError(
                            "AI returned an ambiguous or invalid "
                            "Speaking score. "
                            f"Guardrail status: "
                            f"{_speaking_guard_status}. "
                            "Please try Analyze My English again."
                        )

                    speaking_score = int(
                        round(
                            float(
                                _speaking_score_guard["score"]
                            )
                        )
                    )

                    clean_report = re.sub(
                        r"^\s*OVERALL_SCORE:\s*\d{1,3}\s*",
                        "",
                        report,
                        count=1,
                        flags=re.IGNORECASE
                    ).strip()

                    st.session_state["speaking_report"] = clean_report
                    st.session_state["speaking_score"] = speaking_score

                    save_activity(
                        activity_type="Speaking",
                        activity=topic,
                        score=speaking_score
                    )

                    # STEP 4C/4D - finish retry and preserve its score for comparison.
                    if st.session_state.get("step4_retry_mode", False):
                        st.session_state["step4_retry_result_score"] = speaking_score
                        st.session_state["step4_retry_mode"] = False

                    # Refresh so Adaptive Coach and comparison use the latest result.
                    st.rerun()
                except Exception as e:
                    st.info("🤖 The coach could not analyze this answer right now. Please try again.")

    if st.session_state.get("speaking_report"):
        st.divider()
        st.subheader("📊 Speaking Feedback")

        speaking_score = st.session_state.get("speaking_score")
        if speaking_score is not None:
            st.metric("🎯 Text-Based Speaking Score", f"{speaking_score}%")
            st.caption(
                "This score evaluates grammar, sentence formation, vocabulary, "
                "clarity and topic relevance from your typed answer. "
                "Pronunciation and voice fluency will be added in the Audio stage."
            )

        st.markdown(st.session_state["speaking_report"])

        if st.button("🔄 Clear Speaking Feedback"):
            st.session_state["speaking_report"] = ""
            st.session_state["speaking_score"] = None
            st.rerun()


# ============================================================
# STEP 5 - COMPLETE AI INTERVIEW COACH
# ============================================================

elif page == "🎤 Interviews":

    friendly_coach_message("🎤 Interviews")

    st.title("🎤 AI Interview Coach")
    st.caption("Question → Answer → AI Feedback → Score → Next Question → Final Report")

    # ========================================================
    # STEP 13 - COMPLETE FACE-TO-FACE INTERVIEW COACH
    # Corrected: clean interview video + separate live feedback
    # ========================================================
    interview_mode = st.radio(
        "Practice mode",
        ["Text Practice", "🎙️ Voice Practice", "📷 Face-to-Face Practice"],
        horizontal=True,
        key="step13_interview_mode",
        disabled=st.session_state.get("interview_started", False),
    )

    if st.session_state.get("step13_camera_analyzer") is None:
        st.session_state["step13_camera_analyzer"] = FaceToFaceAnalyzer()

    camera_analyzer = st.session_state["step13_camera_analyzer"]

    if st.session_state.get("step13_live_audio_collector") is None:
        st.session_state["step13_live_audio_collector"] = LiveInterviewAudioCollector()
    live_audio_collector = st.session_state["step13_live_audio_collector"]
    live_ctx = None

    if interview_mode == "📷 Face-to-Face Practice":
        st.subheader("🎥 Live Face-to-Face Interview")
        st.caption(
            "Allow camera and microphone access, then start the live session. "
            "Record each answer using the buttons below the question."
        )

        video_col, feedback_col = st.columns([2.2, 1])

        with video_col:
            st.markdown("#### 👤 LIVE CANDIDATE VIDEO — Keep Camera ON During the Interview")
            try:
                live_ctx = webrtc_streamer(
                    key="step13_face_to_face_live_session",
                    mode=WebRtcMode.SENDRECV,
                    media_stream_constraints={
                        "video": True,
                        "audio": {
                            "echoCancellation": True,
                            "noiseSuppression": True,
                            "autoGainControl": True
                        }
                    },
                    video_frame_callback=camera_analyzer.process,
                    audio_frame_callback=live_audio_collector.process,
                    async_processing=True
                )
            except Exception:
                st.warning("Interview camera could not start. Check browser camera permission.")

        with feedback_col:
            st.markdown("#### 📊 Live Camera Feedback")
            st.caption("🟢 Camera analysis active while this page is open.")

            @st.fragment(run_every="1s")
            def step13_live_camera_panel():
                stats = camera_analyzer.summary()
                if not stats.get("available"):
                    st.info("Waiting for face detection...")
                    return

                st.write(f"**Camera Presence — {stats['camera_presence_score']}%**")
                st.progress(max(0, min(100, stats["camera_presence_score"])) / 100)

                st.write(f"**Face Visible — {stats['presence']}%**")
                st.progress(max(0, min(100, stats["presence"])) / 100)

                st.write(f"**Centered — {stats['centered']}%**")
                st.progress(max(0, min(100, stats["centered"])) / 100)

                st.write(f"**Good Distance — {stats['distance']}%**")
                st.progress(max(0, min(100, stats["distance"])) / 100)

                st.write(f"**Diversions — {stats['diversions']}**")

                if stats["presence"] < 75:
                    st.warning("⚠️ Keep your full face visible.")
                elif stats["centered"] < 70:
                    st.warning("↔️ Move toward the center.")
                elif stats["distance"] < 70:
                    if stats["too_far"] > stats["too_close"]:
                        st.warning("🔎 Move slightly closer.")
                    else:
                        st.warning("↩️ Move slightly back.")
                else:
                    st.success("✅ Good camera position.")

            step13_live_camera_panel()

            st.markdown("#### 👀 Interview Tips")
            st.write("• Look toward the camera naturally.")
            st.write("• Keep your full face visible.")
            st.write("• Sit upright and stay centered.")
            st.write("• Pause briefly before answering.")
            st.write("• Do not rush or make unnecessary side movements.")

        st.info(
            "FACE-TO-FACE SESSION: Keep your camera ON for the full interview. "
            "The AI Interviewer asks questions one by one. "
            "record each answer in this camera session, review its text, get AI feedback, then continue. "
            "Camera behaviour is measured throughout the session."
        )
        st.caption(
            "Camera Presence measures face visibility, centering and approximate distance. "
            "It is not a true eye-contact, attention or emotion score. Video is not saved. "
            "Voice is transcribed after you stop recording, not word by word while speaking."
        )

    interview_type = st.selectbox(
        "Choose interview type",
        ["HR", "Technical", "Technical HR", "Project", "Mock", "Random"],
        key="step5_interview_type",
        disabled=st.session_state.get("interview_started", False)
    )

    difficulty = st.selectbox(
        "Choose difficulty",
        ["Beginner", "Intermediate", "Advanced"],
        key="step5_interview_difficulty",
        disabled=st.session_state.get("interview_started", False)
    )

    # ========================================================
    # INTERVIEW QUESTION BANKS
    # Six independent interview modes with 10 relevant
    # and important questions in each mode.
    # ========================================================

    question_banks = {
        "HR": [
            "Tell me about yourself.",
            "Why are you interested in this role?",
            "Why should we hire you?",
            "What are your key strengths?",
            "What is one area you are currently improving?",
            "Describe a workplace challenge you faced and how you handled it.",
            "Tell me about a time you worked successfully as part of a team.",
            "How do you handle feedback from your manager or colleagues?",
            "How do you prioritize your work when you have multiple responsibilities?",
            "Where do you see yourself in the next five years?"
        ],

        "Technical": [
            "What is Artificial Intelligence, and how is Machine Learning related to it?",
            "What is the difference between regression and classification?",
            "What is overfitting in Machine Learning, and how can you reduce it?",
            "How would you handle missing values in a dataset before training a model?",
            "Explain accuracy, precision, recall and F1 score.",
            "Why do we split data into training and testing datasets?",
            "What is feature selection, and why is it important in Machine Learning?",
            "What is the difference between categorical and numerical data, and how would you preprocess them?",
            "How would you compare multiple Machine Learning models and select an appropriate model?",
            "After building a Machine Learning model, how would you make it available for users in a real-world application?"
        ],

        "Technical HR": [
            "Briefly explain your technical skills and how you have applied them in your work or projects.",
            "Describe a technical problem you solved and the result you achieved.",
            "How do you learn a new technology when you have no previous experience with it?",
            "How would you explain a complex technical concept to a non-technical person?",
            "How do you balance technical quality, deadlines and teamwork?",
            "Tell me about a technical mistake or issue you faced and how you corrected it.",
            "How do you decide which technology or approach to use when solving a problem?",
            "Describe a situation where you worked with others to solve a technical challenge.",
            "How can your existing domain experience support your work with AI and Data Science?",
            "What technical skills are you currently improving, and how will they help your career?"
        ],

        "Project": [
            "Explain one project from the problem statement to the final solution.",
            "What problem were you trying to solve with this project, and why was it important?",
            "What data did you use in the project, and how did you prepare or preprocess it?",
            "What technologies, tools or models did you use, and why did you choose them?",
            "What was the biggest technical challenge in the project, and how did you solve it?",
            "How did you test or evaluate whether the project was working correctly?",
            "How did you convert your analysis or model into a usable application or solution?",
            "What practical or business value does your project provide to the user?",
            "What did you personally learn while developing this project?",
            "What would you improve or add if you developed the project again?"
        ],

        "Mock": [
            "Please introduce yourself and briefly describe your professional background.",
            "Why are you interested in this role and this career direction?",
            "What are the most relevant skills you can bring to this role?",
            "Explain one project or professional achievement you are proud of.",
            "Describe a difficult problem you solved under pressure.",
            "Tell me about a situation where teamwork was important to achieve a result.",
            "What are your key strengths, and how have they helped you professionally?",
            "Tell me about a time you had to learn or adapt to something new.",
            "What are your career goals for the next few years?",
            "Why should we consider you for this position?"
        ],

        "Random": [
            "What is one skill you have improved recently, and how did you improve it?",
            "Describe a situation where you had to adapt quickly to a change.",
            "Explain a technical concept you know in simple words.",
            "What is one professional achievement that taught you an important lesson?",
            "If you do not know an answer in an interview, how would you handle the situation?",
            "Tell me about a time you received feedback and used it to improve.",
            "How do you stay organized when you have several tasks to complete?",
            "Describe a problem where you had to think carefully before making a decision.",
            "What motivates you to continue learning new skills?",
            "What value would you like to bring to your future team or organization?"
        ]
    }

    # --------------------------------------------------------
    # Interview length
    # Quick Interview = 5 questions
    # Full Interview  = 10 questions
    # --------------------------------------------------------

    interview_length = st.radio(
        "Choose interview length",
        ["Quick Interview - 5 Questions", "Full Interview - 10 Questions"],
        horizontal=True,
        key="step5_interview_length",
        disabled=st.session_state.get("interview_started", False)
    )

    selected_question_count = (
        5
        if interview_length == "Quick Interview - 5 Questions"
        else 10
    )


    if not st.session_state.get("interview_started", False):
        st.info(
            "Choose the interview type and difficulty, then start. "
            "The coach evaluates your answer text for English and interview effectiveness. "
            "Face-to-Face mode adds live camera practice."
        )

        if st.button(
            "🚀 Start Interview",
            type="primary",
            use_container_width=True,
            key="step5_start_interview"
        ):
            st.session_state.interview_started = True
            st.session_state["step5_active_interview_type"] = interview_type
            st.session_state["step5_active_difficulty"] = difficulty
            st.session_state["step5_active_question_count"] = selected_question_count

            # Clear previous Interview Intelligence session
            st.session_state.pop(
                "interview_intelligence_result",
                None
            )

            st.session_state.pop(
                "interview_intelligence_question",
                None
            )

            st.session_state.pop(
                "interview_intelligence_answer",
                None
            )

            st.session_state.pop(
                "interview_intelligence_score",
                None
            )
            st.session_state.interview_count = 0
            st.session_state.interview_history = []
            st.session_state.interview_feedback = ""
            st.session_state.final_interview_report = ""
            st.session_state["step5_interview_scores"] = []
            st.session_state["step5_interview_reports"] = []
            st.session_state["step5_interview_questions"] = []
            st.session_state["step13_last_camera_summary"] = None
            live_audio_collector.clear()
            for _i in range(10):
                st.session_state.pop(f"step13_voice_transcript_{_i}", None)
                st.session_state.pop(f"step13_transcript_{_i}", None)
                st.session_state.pop(f"step5_answer_{_i}", None)
                st.session_state.pop(f"interview_answer_{_i}", None)
                st.session_state.pop(f"step5_answer_editor_nonce_{_i}", None)
                st.session_state.pop(f"step13_live_recording_{_i}", None)
                st.session_state.pop(f"step5_audio_hash_{_i}", None)
                st.session_state.pop(f"step5_audio_nonce_{_i}", None)
            if interview_mode == "📷 Face-to-Face Practice":
                camera_analyzer.reset()

            questions = question_banks[interview_type]
            st.session_state.interview_question = questions[0]
            st.rerun()

    if st.session_state.get("interview_started", False):
        active_interview_type = st.session_state.get("step5_active_interview_type", interview_type)
        active_difficulty = st.session_state.get("step5_active_difficulty", difficulty)
        questions = question_banks[active_interview_type]
        st.caption(f"Active interview: {active_interview_type} • {active_difficulty}")
        count = st.session_state.get("interview_count", 0)
        active_question_count = st.session_state.get(
            "step5_active_question_count",
            selected_question_count
        )
        total_questions = min(active_question_count, len(questions))

        st.progress(min(count, total_questions) / total_questions)
        st.caption(f"Question {min(count + 1, total_questions)} of {total_questions}")

        question = st.session_state.get("interview_question", "")
        if not question:
            question = questions[count % len(questions)]
            st.session_state.interview_question = question

        if interview_mode == "📷 Face-to-Face Practice":
            st.markdown("### 🧑‍💼 AI Interviewer")
            interviewer_box, status_box = st.columns([3, 1])
            with interviewer_box:
                st.info(f"**Question {count + 1}:** {question}")
            with status_box:
                st.success("🔴 LIVE INTERVIEW")
            st.caption(
                "Your camera remains live while you answer. Camera feedback is for practice only; this app does not save your interview video."
            )
        else:
            st.info(f"🎤 {question}")

        # Voice practice uses the same reliable recorder as the Speaking page.
        if interview_mode == "🎙️ Voice Practice":
            st.markdown("#### 🎙️ Record Your Answer")
            st.caption(
                "Press the microphone, speak, then stop recording. "
                "Your editable transcript appears below."
            )
            audio_nonce = st.session_state.get(f"step5_audio_nonce_{count}", 0)
            recorded_answer = st.audio_input(
                "Speak your answer",
                key=f"step5_voice_audio_{count}_{audio_nonce}",
            )
            if recorded_answer is not None:
                audio_bytes = recorded_answer.getvalue()
                audio_hash = hashlib.sha256(audio_bytes).hexdigest()
                hash_key = f"step5_audio_hash_{count}"
                if audio_hash != st.session_state.get(hash_key):
                    if sr is None:
                        st.warning("Speech recognition is unavailable. You can type your answer below.")
                    elif not audio_bytes:
                        st.warning("The microphone recording was empty. Please try again.")
                    else:
                        try:
                            with st.spinner("Converting your answer to text..."):
                                recognizer = sr.Recognizer()
                                with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                                    speech = recognizer.record(source)
                                transcript = recognizer.recognize_google(
                                    speech, language="en-IN"
                                ).strip()
                            if transcript and voice_transcript_is_english(transcript):
                                st.session_state[hash_key] = audio_hash
                                st.session_state[f"step13_transcript_{count}"] = transcript
                                st.session_state[f"step5_answer_{count}"] = transcript
                                st.session_state[f"step5_answer_editor_nonce_{count}"] = (
                                    st.session_state.get(f"step5_answer_editor_nonce_{count}", 0) + 1
                                )
                                st.rerun()
                            elif transcript:
                                show_english_voice_message()
                            else:
                                st.info("I could not hear the answer clearly. Please record again.")
                        except sr.UnknownValueError:
                            st.info("I could not understand the recording. Please try again or type below.")
                        except sr.RequestError:
                            st.warning("Speech recognition is unavailable. Please try again or type below.")
                        except Exception as exc:
                            st.warning(
                                "The recording could not be converted. "
                                "Please retry or type your answer below. "
                                f"({type(exc).__name__})"
                            )

            transcript = st.session_state.get(f"step13_transcript_{count}", "")
            if transcript:
                st.success("Transcript ready. Review it in the answer box below.")

            if st.button(
                "🔄 Record Again", key=f"step5_voice_retry_{count}",
                use_container_width=True,
            ):
                st.session_state[f"step5_audio_nonce_{count}"] = audio_nonce + 1
                for key in (
                    f"step5_audio_hash_{count}", f"step13_transcript_{count}",
                    f"step5_answer_{count}", f"interview_answer_{count}",
                ):
                    st.session_state.pop(key, None)
                st.session_state[f"step5_answer_editor_nonce_{count}"] = (
                    st.session_state.get(f"step5_answer_editor_nonce_{count}", 0) + 1
                )
                st.rerun()

        if interview_mode == "📷 Face-to-Face Practice":
            st.markdown("#### 🎙️ Your Live Camera Answer")
            st.caption(
                "Start the camera above and allow microphone access. Press Start Answer "
                "Recording, speak to the camera, then press Stop & Transcribe. "
                "The words will appear in the answer box below."
            )
            recording_key = f"step13_live_recording_{count}"
            camera_running = bool(live_ctx and live_ctx.state.playing)
            if not camera_running:
                st.info("Start the camera and microphone session above to record your answer.")
                if st.session_state.get(recording_key):
                    live_audio_collector.clear()
                    st.session_state[recording_key] = False

            start_col, stop_col = st.columns(2)
            with start_col:
                start_capture = st.button(
                    "🎙️ Start Answer Recording",
                    key=f"step13_start_capture_{count}",
                    use_container_width=True,
                    disabled=not camera_running or st.session_state.get(recording_key, False),
                )
            with stop_col:
                stop_capture = st.button(
                    "⏹️ Stop & Transcribe",
                    key=f"step13_stop_capture_{count}",
                    use_container_width=True,
                    disabled=not camera_running or not st.session_state.get(recording_key, False),
                )

            if start_capture:
                try:
                    live_audio_collector.start()
                    st.session_state[f"step13_capture_result_{count}"] = ""
                    st.session_state[recording_key] = True
                    for key in (
                        f"step13_transcript_{count}", f"step5_answer_{count}",
                        f"interview_answer_{count}",
                    ):
                        st.session_state.pop(key, None)
                    st.session_state[f"step5_answer_editor_nonce_{count}"] = (
                        st.session_state.get(f"step5_answer_editor_nonce_{count}", 0) + 1
                    )
                    st.rerun()
                except Exception as exc:
                    st.warning(f"Microphone could not start ({type(exc).__name__}).")

            if st.session_state.get(recording_key, False):
                st.info("🔴 Recording this answer. Speak clearly, then press Stop & Transcribe.")

                @st.fragment(run_every="1s")
                def show_interview_mic_status():
                    frames, total, error = live_audio_collector.status()
                    if error:
                        st.error(error)
                    elif frames:
                        st.caption(f"Microphone audio received: {total // 32000} seconds (approximately).")
                    else:
                        st.warning("Waiting for microphone audio. Check browser microphone permission and input device.")

                show_interview_mic_status()

            if stop_capture:
                st.session_state[recording_key] = False
                try:
                    frames, total, capture_error = live_audio_collector.status()
                    if frames == 0 or total == 0:
                        raise RuntimeError(
                            "Camera opened, but no microphone audio reached the app. "
                            "Allow microphone permission in the browser, check its selected input device, "
                            "then restart the camera session."
                        )
                    audio_bytes = live_audio_collector.stop_wav()
                    with st.spinner("Converting your camera interview answer to text..."):
                        transcript, transcription_service = transcribe_interview_audio(audio_bytes)
                    if transcript and english_practice_language_check(transcript)[0]:
                        st.session_state[f"step13_transcript_{count}"] = transcript
                        st.session_state[f"step5_answer_{count}"] = transcript
                        st.session_state[f"step5_answer_editor_nonce_{count}"] = (
                            st.session_state.get(f"step5_answer_editor_nonce_{count}", 0) + 1
                        )
                        st.session_state[f"step13_capture_result_{count}"] = (
                            f"Transcript ready ({transcription_service}). Review it below; "
                            "you can correct any missing words in the answer box."
                        )
                        st.rerun()
                    elif transcript:
                        st.session_state[f"step13_capture_result_{count}"] = (
                            "Audio was recognized, but the response was not in English. Please try again in English."
                        )
                        show_english_voice_message()
                    else:
                        st.session_state[f"step13_capture_result_{count}"] = "No clear words were recognized. Please record again."
                        st.info("No clear words were recognized. Please record again.")
                except Exception as exc:
                    st.session_state[f"step13_capture_result_{count}"] = (
                        f"Audio capture / transcription issue: {type(exc).__name__}: {exc}"
                    )
                    if sr is not None and isinstance(exc, sr.UnknownValueError):
                        st.info("I could not understand the recording. Please try again.")
                    elif sr is not None and isinstance(exc, sr.RequestError):
                        st.warning("Speech recognition is unavailable. Try again shortly.")
                    else:
                        st.warning(
                            "Camera interview audio could not be converted. "
                            f"{str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__}"
                        )
                finally:
                    live_audio_collector.clear()

            capture_result = st.session_state.get(f"step13_capture_result_{count}")
            if capture_result:
                st.info(capture_result)

            if st.session_state.get(f"step13_transcript_{count}"):
                st.markdown("**Recognized words:**")
                st.write(st.session_state[f"step13_transcript_{count}"])

        answer_nonce = st.session_state.get(f"step5_answer_editor_nonce_{count}", 0)
        answer_key = f"interview_answer_{count}_{answer_nonce}"
        if answer_key not in st.session_state:
            st.session_state[answer_key] = st.session_state.get(
                f"step5_answer_{count}", ""
            )
        answer = st.text_area(
            "Your interview answer / voice transcript",
            height=180,
            key=answer_key,
            placeholder=(
                "Type your answer, or record it in Voice / Face-to-Face mode. "
                "The transcript is editable before AI analysis."
            )
        )


        # ================================================
        # INTERVIEW INTELLIGENCE V2 DISPLAY
        # ================================================

        _ii_display = st.session_state.get(
            "interview_intelligence_result"
        )

        _ii_saved_question = st.session_state.get(
            "interview_intelligence_question",
            ""
        )

        # Display the latest completed-answer intelligence
        # after Streamlit advances to the next question.
        #
        # The saved question intentionally belongs to the
        # PREVIOUS completed answer, while `question` may now
        # contain the NEXT interview question after st.rerun().
        if _ii_display:

            st.markdown("---")

            st.subheader(
                "🧠 Interview Intelligence"
            )

            if _ii_saved_question:

                st.caption(
                    "Analysis for your previous answer:"
                )

                st.markdown(
                    "**"
                    + str(_ii_saved_question)
                    + "**"
                )

            _ii_c1, _ii_c2, _ii_c3, _ii_c4 = (
                st.columns(4)
            )

            _ii_c1.metric(
                "AI Score",
                str(
                    _ii_display.get(
                        "overall_score",
                        0
                    )
                )
                + "/100"
            )

            _ii_c2.metric(
                "Relevance",
                str(
                    _ii_display.get(
                        "relevance_score",
                        0
                    )
                )
                + "/100"
            )

            _ii_c3.metric(
                "Structure",
                str(
                    _ii_display.get(
                        "structure_score",
                        0
                    )
                )
                + "/100"
            )

            _ii_c4.metric(
                "Clarity",
                str(
                    _ii_display.get(
                        "clarity_score",
                        0
                    )
                )
                + "/100"
            )

            st.caption(
                "Overall score source: "
                + str(
                    _ii_display.get(
                        "overall_score_source",
                        "Interview AI"
                    )
                )
            )

            st.caption(
                "Question type: "
                + str(
                    _ii_display.get(
                        "question_type",
                        "General"
                    )
                )
                + "  •  Next difficulty: "
                + str(
                    _ii_display.get(
                        "next_difficulty",
                        active_difficulty
                    )
                )
            )

            # --------------------------------------------
            # Strengths
            # --------------------------------------------

            _ii_strengths = (
                _ii_display.get(
                    "strengths",
                    []
                )
            )

            if _ii_strengths:

                st.markdown(
                    "**✅ What worked well**"
                )

                for _ii_item in _ii_strengths:

                    st.write(
                        "• " + str(_ii_item)
                    )


            # --------------------------------------------
            # Improvements
            # --------------------------------------------

            _ii_weaknesses = (
                _ii_display.get(
                    "weaknesses",
                    []
                )
            )

            if _ii_weaknesses:

                st.markdown(
                    "**🎯 What to improve**"
                )

                for _ii_item in _ii_weaknesses:

                    st.write(
                        "• " + str(_ii_item)
                    )


            # --------------------------------------------
            # Missing points
            # --------------------------------------------

            _ii_missing = (
                _ii_display.get(
                    "missing_points",
                    []
                )
            )

            if _ii_missing:

                st.markdown(
                    "**📌 Missing point**"
                )

                for _ii_item in _ii_missing:

                    st.write(
                        "• " + str(_ii_item)
                    )


            # --------------------------------------------
            # Coach tip
            # --------------------------------------------

            _ii_tip = _ii_display.get(
                "coach_tip",
                ""
            )

            if _ii_tip:

                st.info(
                    "💡 Coach Tip: "
                    + str(_ii_tip)
                )


            # --------------------------------------------
            # Follow-up
            # --------------------------------------------

            _ii_followup = (
                _ii_display.get(
                    "follow_up_question"
                )
            )

            if (
                _ii_display.get(
                    "follow_up_needed",
                    False
                )
                and _ii_followup
            ):

                st.markdown(
                    "**🔁 Smart Follow-up**"
                )

                st.write(
                    str(_ii_followup)
                )


            # --------------------------------------------
            # Session Intelligence
            # --------------------------------------------

            st.markdown(
                "**📈 Session Intelligence**"
            )

            _ii_s1, _ii_s2, _ii_s3 = (
                st.columns(3)
            )

            _ii_s1.metric(
                "Attempts",
                _ii_display.get(
                    "session_attempts",
                    0
                )
            )

            _ii_s2.metric(
                "Average",
                str(
                    _ii_display.get(
                        "session_average",
                        0
                    )
                )
                + "%"
            )

            _ii_s3.metric(
                "Trend",
                str(
                    _ii_display.get(
                        "session_trend",
                        "Not Enough Data"
                    )
                )
            )


            # --------------------------------------------
            # Next action
            # --------------------------------------------

            _ii_next_action = (
                _ii_display.get(
                    "next_action",
                    ""
                )
            )

            if _ii_next_action:

                st.success(
                    "Next Action: "
                    + str(_ii_next_action)
                )

            st.markdown("---")

        ia1, ia2 = st.columns(2)
        with ia1:
            if st.button("🗑️ Clear Current Answer", key=f"step14_clear_interview_{count}", use_container_width=True):
                st.session_state.pop(answer_key, None)
                st.session_state.pop(f"step5_answer_{count}", None)
                st.session_state.pop(f"step13_transcript_{count}", None)
                st.rerun()
        with ia2:
            if st.button("💡 Help Me Answer in English", key=f"step14_hint_interview_{count}", use_container_width=True):
                st.info("Start with one simple English sentence. Then add one reason or example. For experience questions, use Situation → Action → Result.")

        if interview_mode != "Text Practice":
            st.caption(
                "AI feedback uses the editable transcript above. "
                "Review it, then press Analyze Answer."
            )

        c1, c2 = st.columns(2)

        with c1:
            submit_answer = st.button(
                "🤖 Analyze Answer",
                type="primary",
                use_container_width=True,
                key=f"step5_analyze_{count}"
            )

        with c2:
            stop_interview = st.button(
                "⏹️ Finish Interview",
                use_container_width=True,
                key=f"step5_finish_{count}"
            )

        if submit_answer:
            if not answer.strip():
                st.info("🤝 Add a short answer first — even 2 or 3 simple sentences are enough to begin.")
            elif not show_english_retry_support(answer):
                pass
            else:
                system_prompt = """
You are a supportive English Interview Coach.

Evaluate ONLY the learner's written interview answer.
Do not score pronunciation, accent, voice quality, speaking speed,
eye contact, facial expression, or body language.

Return the first line exactly:
OVERALL_SCORE: <integer 0-100>

Then return clean Markdown using exactly these sections:

## Score Breakdown
- Answer Relevance: <0-20>/20
- Clarity and Structure: <0-20>/20
- Grammar: <0-20>/20
- Vocabulary: <0-20>/20
- Interview Effectiveness: <0-20>/20

## What You Did Well
Give 2-3 specific positive points.

## Improve This
Give the most important improvements.

## Better Interview Answer
Rewrite the learner's answer in natural, simple professional English.
Keep the learner's original facts. Do not invent experience or achievements.

## Coach Tips
Give 3 short practical interview tips.

Be supportive, practical and concise.
"""

                user_prompt = f"""
Interview type: {interview_type}
Difficulty: {difficulty}
Practice mode: {interview_mode}
Question: {question}

Learner's answer:
{answer}

Evaluate this answer now.

Important: Evaluate only the written answer. Do not infer or score camera
behavior, appearance, eye contact, facial expression, body language,
pronunciation, accent, voice quality or speaking speed.
"""

                with st.spinner("🤖 Interview coach is analyzing..."):
                    try:
                        report = ask_ai(system_prompt, user_prompt)

                        if not report or not report.strip():
                            raise ValueError("AI returned an empty response.")

                        score_match = re.search(
                            r"OVERALL_SCORE:\s*(\d{1,3})",
                            report,
                            flags=re.IGNORECASE
                        )

                        if not score_match:
                            retry_prompt = system_prompt + """
IMPORTANT: Your previous response missed the required first line.
Start exactly with:
OVERALL_SCORE: <0-100>
"""
                            report = ask_ai(retry_prompt, user_prompt)
                            score_match = re.search(
                                r"OVERALL_SCORE:\s*(\d{1,3})",
                                report or "",
                                flags=re.IGNORECASE
                            )

                        if not score_match:
                            raise ValueError("AI score format was missing. Please try again.")

                        # ====================================
                        # INTERVIEW SCORE SCALE GUARDRAIL V1
                        # ====================================
                        #
                        # The existing Interview AI remains
                        # responsible for generating the score.
                        #
                        # This layer only validates that the
                        # returned value is safe for FluentPath's
                        # expected 0-100 scoring contract.
                        #
                        # It does NOT convert 8 -> 80.
                        # Camera / Mic / STT are untouched.
                        # ====================================

                        _raw_interview_score = score_match.group(1)

                        _score_guard = validate_score_contract(
                            _raw_interview_score
                        )

                        if not _score_guard.get("valid"):
                            _guard_status = _score_guard.get(
                                "status",
                                "UNKNOWN"
                            )

                            raise ValueError(
                                "AI returned an ambiguous or invalid "
                                "Interview score. "
                                f"Guardrail status: {_guard_status}. "
                                "Please try again."
                            )

                        score = int(
                            round(
                                float(
                                    _score_guard["score"]
                                )
                            )
                        )
                        clean_report = re.sub(
                            r"^\s*OVERALL_SCORE:\s*\d{1,3}\s*",
                            "",
                            report,
                            count=1,
                            flags=re.IGNORECASE
                        ).strip()

                        st.session_state["step5_interview_scores"].append(score)
                        st.session_state["step5_interview_reports"].append(clean_report)
                        st.session_state["step5_interview_questions"].append(question)
                        st.session_state.interview_feedback = clean_report

                        # ====================================
                        # INTERVIEW INTELLIGENCE V2 FINAL
                        # ====================================
                        #
                        # Existing AI score remains the
                        # primary interview score.
                        #
                        # V2 adds:
                        # - structural analysis
                        # - weakness detection
                        # - safe follow-up
                        # - adaptive difficulty
                        # - session intelligence
                        #
                        # Camera / Mic / STT are untouched.
                        # ====================================

                        try:

                            _ii_recent_scores = list(
                                st.session_state.get(
                                    "step5_interview_scores",
                                    []
                                )
                            )

                            # Current score was just appended
                            # above. Remove it from history
                            # before passing recent scores,
                            # because V2 adds current score
                            # internally for session analysis.
                            if _ii_recent_scores:

                                _ii_recent_scores = (
                                    _ii_recent_scores[:-1]
                                )

                            _ii_result = (
                                run_interview_intelligence(
                                    mode=active_interview_type,
                                    level=active_difficulty,
                                    question=question,
                                    answer=answer,
                                    recent_scores=_ii_recent_scores,
                                    existing_score=score,
                                )
                            )

                            # Save result for display after
                            # Streamlit reruns.
                            st.session_state[
                                "interview_intelligence_result"
                            ] = _ii_result

                            # Keep result tied to the exact
                            # analyzed question.
                            st.session_state[
                                "interview_intelligence_question"
                            ] = question

                            st.session_state[
                                "interview_intelligence_answer"
                            ] = answer

                            st.session_state[
                                "interview_intelligence_score"
                            ] = score

                        except Exception:

                            # Intelligence V2 must never break
                            # the existing interview workflow.
                            st.session_state[
                                "interview_intelligence_result"
                            ] = None

                        st.session_state.interview_history.append(
                            {
                                "question": question,
                                "answer": answer,
                                "score": score
                            }
                        )

                        save_activity(
                            activity_type="Interview",
                            activity=f"{active_interview_type} - {question[:120]}",
                            score=score,
                            review_data={
                                "title": f"{active_interview_type} Interview",
                                "interview_type": active_interview_type,
                                "difficulty": active_difficulty,
                                "mode": interview_mode,
                                "question": question,
                                "your_answer": answer,
                                "ai_feedback": clean_report,
                                "score": score,
                                "interview_intelligence": (
                                    st.session_state.get(
                                        "interview_intelligence_result"
                                    )
                                ),
                                "question_number": count + 1,
                                "total_questions": total_questions
                            }
                        )

                        next_count = count + 1
                        st.session_state.interview_count = next_count

                        if next_count >= total_questions:
                            st.session_state.interview_started = False
                            if interview_mode == "📷 Face-to-Face Practice":
                                st.session_state["step13_last_camera_summary"] = camera_analyzer.summary()
                                live_audio_collector.clear()
                                for _i in range(10):
                                    st.session_state.pop(f"step13_live_recording_{_i}", None)
                                    st.session_state.pop(f"step5_audio_hash_{_i}", None)
                                    st.session_state.pop(f"step5_audio_nonce_{_i}", None)
                        else:
                            st.session_state.interview_question = questions[
                                next_count % len(questions)
                            ]

                        st.rerun()

                    except Exception as e:
                        st.info("🤖 Interview feedback is temporarily unavailable. Your answer is still here — please try Analyze Answer again.")

        if stop_interview:
            st.session_state.interview_started = False
            if interview_mode == "📷 Face-to-Face Practice":
                st.session_state["step13_last_camera_summary"] = camera_analyzer.summary()
                live_audio_collector.clear()
                for _i in range(10):
                    st.session_state.pop(f"step13_live_recording_{_i}", None)
                st.session_state.pop(f"step5_audio_hash_{_i}", None)
                st.session_state.pop(f"step5_audio_nonce_{_i}", None)
            st.rerun()

    scores = st.session_state.get("step5_interview_scores", [])
    reports = st.session_state.get("step5_interview_reports", [])
    asked_questions = st.session_state.get("step5_interview_questions", [])

    camera_stats = st.session_state.get("step13_last_camera_summary")

    if camera_stats:
        st.divider()
        st.subheader("📷 Face-to-Face Interview Feedback")

        if camera_stats.get("available"):
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Camera Presence", f"{camera_stats['camera_presence_score']}%")
            f2.metric("Face Visible", f"{camera_stats['presence']}%")
            f3.metric("Centered", f"{camera_stats['centered']}%")
            f4.metric("Good Distance", f"{camera_stats['distance']}%")

            st.markdown("#### 🎯 Final Camera Feedback")
            st.info(face_to_face_feedback(camera_stats))

            st.caption(
                f"Measured camera-position diversions: {camera_stats['diversions']}. "
                "This is not an eye-contact or attention score."
            )

            st.markdown("#### 👀 Coach Tips")
            if camera_stats["presence"] < 90:
                st.write("• Keep your complete face visible throughout the answer.")
            if camera_stats["centered"] < 80:
                st.write("• Reduce side movement and stay closer to the center.")
            if camera_stats["distance"] < 80:
                st.write("• Adjust your chair/camera distance before starting.")
            st.write("• Look toward the camera naturally when speaking.")
            st.write("• Sit upright and keep a relaxed expression.")
            st.write("• Pause briefly before answering; do not rush.")
            st.write("• Keep your answer structured and avoid unnecessary diversions.")
        else:
            st.warning(
                "No measurable camera frames were captured. "
                "Keep the live camera running during the interview and try again."
            )

    if scores:
        st.divider()
        st.subheader("📊 Interview Progress")

        avg_score = round(sum(scores) / len(scores))
        best_score = max(scores)
        latest_score = scores[-1]

        m1, m2, m3 = st.columns(3)
        m1.metric("Latest Score", f"{latest_score}%")
        m2.metric("Average Score", f"{avg_score}%")
        m3.metric("Best Score", f"{best_score}%")

        if reports:
            st.markdown("### 🧑‍🏫 Latest Coach Feedback")
            st.markdown(reports[-1])

    if scores and not st.session_state.get("interview_started", False):
        st.divider()
        st.subheader("🏁 Final Interview Report")

        avg_score = round(sum(scores) / len(scores))

        if avg_score >= 80:
            level_message = (
                "Strong practice performance. Continue improving answer depth "
                "and consistency."
            )
        elif avg_score >= 60:
            level_message = (
                "Good progress. Focus on structure, grammar and stronger examples."
            )
        else:
            level_message = (
                "Keep practicing with short structured answers before adding detail."
            )

        st.metric("Overall Interview Practice Score", f"{avg_score}%")
        st.info(level_message)

        st.markdown("### Question Summary")
        for idx, (q, s) in enumerate(zip(asked_questions, scores), start=1):
            st.write(f"**Q{idx}. {q}** — {s}%")

        st.markdown("### 🎯 Interview Coach Reminder")
        st.write(
            "Use a simple structure: direct answer → reason → example → result. "
            "For behavioral questions, STAR (Situation, Task, Action, Result) can help."
        )

        if st.button(
            "🔄 Start New Interview",
            use_container_width=True,
            key="step5_new_interview"
        ):
            st.session_state.interview_started = False
            st.session_state.interview_question = ""
            st.session_state.interview_history = []
            st.session_state.interview_feedback = ""
            st.session_state.interview_count = 0
            st.session_state.final_interview_report = ""
            st.session_state["step5_interview_scores"] = []
            st.session_state["step5_interview_reports"] = []
            st.session_state["step5_interview_questions"] = []
            st.rerun()



# ============================================================
# TECHPREP - TECHNICAL INTERVIEW PREPARATION
# ============================================================

elif page == "🧑‍💻 TechPrep":

    friendly_coach_message("🧑‍💻 TechPrep")

    st.title("🧑‍💻 TechPrep")

    # ========================================================
    # ADAPTIVE LEARNING PLAN
    # Uses the logged-in learner's real TechPrep score history.
    # ========================================================

    try:
        adaptive_user_id = st.session_state.get("user_id")

        if adaptive_user_id:

            adaptive_data = build_user_adaptive_profile(
                adaptive_user_id
            )

            adaptive_profiles = adaptive_data.get(
                "profiles",
                {}
            )

            adaptive_next = adaptive_data.get(
                "next_recommendation",
                {}
            )

            with st.expander(
                "🧠 Your Adaptive Learning Plan",
                expanded=True
            ):

                if adaptive_profiles:

                    recommended_topic = adaptive_next.get(
                        "topic"
                    )

                    selected_profile = adaptive_profiles.get(
                        recommended_topic
                    )

                    if selected_profile is None:
                        selected_profile = min(
                            adaptive_profiles.values(),
                            key=lambda item: item.get(
                                "average_score",
                                0
                            )
                        )

                    st.markdown(
                        f"**Recommended Focus:** "
                        f"{adaptive_next.get('topic', 'TechPrep')}"
                    )

                    st.caption(
                        adaptive_next.get(
                            "reason",
                            "Continue practising to build "
                            "your learning profile."
                        )
                    )

                    ac1, ac2, ac3, ac4 = st.columns(4)

                    ac1.metric(
                        "Average Score",
                        f"{selected_profile.get('average_score', 0)}%"
                    )

                    ac2.metric(
                        "Current Level",
                        selected_profile.get(
                            "level",
                            "Beginner"
                        )
                    )

                    ac3.metric(
                        "Trend",
                        selected_profile.get(
                            "trend",
                            "Not Enough Data"
                        )
                    )

                    ac4.metric(
                        "Next Difficulty",
                        selected_profile.get(
                            "recommended_difficulty",
                            "Beginner"
                        )
                    )

                    st.info(
                        "🎯 Next Action: "
                        + selected_profile.get(
                            "recommended_action",
                            "Continue practising."
                        )
                    )

                    st.caption(
                        f"Based on "
                        f"{selected_profile.get('attempts', 0)} "
                        f"scored TechPrep attempt(s)."
                    )

                else:

                    st.info(
                        "Complete your first scored TechPrep "
                        "practice to activate personalized "
                        "learning recommendations."
                    )

    except Exception:
        # Adaptive recommendations must never block TechPrep.
        pass

    st.caption(
        "Know → Understand → Structure → Say → Improve"
    )

    st.info(
        "Choose a technical topic and practise important interview questions. "
        "Understand the concept first, then build a clear interview answer."
    )

    # --------------------------------------------------------
    # QUESTION BANKS
    # --------------------------------------------------------

    # TECHPREP DYNAMIC 20 QUESTION SYSTEM V1
    techprep_banks = {
        'Python': [
            'What is Python and why is it widely used?',
            'What is the difference between a list, tuple, set and dictionary in Python?',
            'What is the difference between mutable and immutable objects in Python?',
            'What is the difference between == and is in Python?',
            'What are *args and **kwargs in Python?',
            'What is list comprehension and when would you use it?',
            'How does exception handling work in Python?',
            'Explain functions and lambda functions in Python.',
            'What are the main OOP concepts in Python?',
            'What is the difference between an iterator and a generator?',
            'What is the difference between append() and extend() in Python?',
            'What is the difference between / and // in Python?',
            'What is the difference between return and print() in a Python function?',
            'How do decorators work in Python and when would you use them?',
            'What is the difference between shallow copy and deep copy in Python?',
            'How does Python manage memory and garbage collection?',
            'What is the difference between a module and a package in Python?',
            'How would you read, process and write a large file efficiently in Python?',
            'How would you improve the performance of slow Python code?',
            'If a Python program fails in production, how would you debug the problem?',
        ],
        'Machine Learning': [
            'What is Machine Learning and how is it related to Artificial Intelligence?',
            'What is the difference between supervised and unsupervised learning?',
            'What is the difference between regression and classification?',
            'What are overfitting and underfitting, and how can they be handled?',
            'Why do we split data into training, validation and testing datasets?',
            'What is the bias-variance tradeoff?',
            'What are feature engineering and feature selection?',
            'How would you handle missing values, outliers and feature scaling?',
            'Explain accuracy, precision, recall and F1 score.',
            'What is cross-validation and how do you select the best model?',
            'What is a confusion matrix and what information does it provide?',
            'What is the difference between bagging and boosting?',
            'What is regularization and what is the difference between L1 and L2 regularization?',
            'How would you handle an imbalanced classification dataset?',
            'What is hyperparameter tuning and which techniques can be used for it?',
            'What is data leakage and how can you prevent it?',
            'How do you decide whether a feature should be removed from a Machine Learning model?',
            'How would you determine whether a model is overfitting using training and validation results?',
            'If two models have similar accuracy, how would you decide which model to deploy?',
            'After deploying a Machine Learning model, how would you monitor whether its performance is degrading?',
        ],
        'Data Science': [
            'What is Data Science and what are the main stages of a Data Science project?',
            'What is Exploratory Data Analysis and why is it important?',
            'How do you clean a real-world dataset before analysis?',
            'What is the difference between structured and unstructured data?',
            'How would you handle missing values in a dataset?',
            'How do you identify and handle outliers?',
            'What is feature engineering and why is it useful?',
            'What is the difference between correlation and causation?',
            'How do you evaluate whether a Data Science solution is performing well?',
            'How would you communicate Data Science insights to a non-technical stakeholder?',
            'What is the difference between univariate, bivariate and multivariate analysis?',
            'What is the difference between numerical and categorical data?',
            'How would you encode categorical variables for analysis or Machine Learning?',
            'Why is feature scaling required for some algorithms but not for others?',
            'What is multicollinearity and how would you detect it?',
            'How do you decide which visualisation is appropriate for a particular dataset?',
            'What is data leakage and why can it make model performance misleading?',
            'How would you approach a dataset in which several columns have a large percentage of missing values?',
            'How would you validate whether an insight found in your data is meaningful and not just accidental?',
            'If a Data Science solution performs well technically but provides little business value, what would you do?',
        ],
        'Statistics': [
            'What is the difference between mean, median and mode?',
            'What are variance and standard deviation?',
            'What is probability and why is it important in Data Science?',
            'What is a normal distribution?',
            'What is the difference between a population and a sample?',
            'What is hypothesis testing?',
            'What is a p-value and how is it interpreted?',
            'What is a confidence interval?',
            'What is the difference between covariance and correlation?',
            'What are Type I and Type II errors?',
            'What is the difference between descriptive and inferential statistics?',
            'What are skewness and kurtosis?',
            'What is the Central Limit Theorem and why is it useful?',
            'What is statistical significance?',
            'What is the difference between a one-tailed and a two-tailed hypothesis test?',
            'When would you use a t-test, chi-square test and ANOVA?',
            'What is the difference between a paired t-test and an independent t-test?',
            'What assumptions should be checked before applying common statistical tests?',
            'How would you determine whether an extreme value is a genuine observation or an outlier?',
            'If a result is statistically significant, does it always mean it is practically important? Explain.',
        ],
        'SQL': [
            'What is SQL and what is a relational database?',
            'What is the difference between WHERE and HAVING?',
            'Explain INNER JOIN, LEFT JOIN, RIGHT JOIN and FULL JOIN.',
            'What is GROUP BY and how is it used with aggregate functions?',
            'What is the difference between a subquery and a CTE?',
            'What are primary keys and foreign keys?',
            'What is database normalization?',
            'What are SQL window functions?',
            'What is an index and how can it improve query performance?',
            'How would you find duplicate records in a SQL table?',
            'What is the difference between DELETE, TRUNCATE and DROP?',
            'What is the difference between UNION and UNION ALL?',
            'What is the difference between ROW_NUMBER(), RANK() and DENSE_RANK()?',
            'What are transactions and the ACID properties in a database?',
            'What is the difference between clustered and non-clustered indexes?',
            'How would you find the second highest value in a SQL table?',
            'How would you find the top N records within each category using SQL?',
            'How would you calculate a running total using a SQL window function?',
            'A SQL query is running slowly on a large table. How would you investigate and optimise it?',
            'How would you design a query to compare current-month performance with previous-month performance?',
        ],
        'Power BI': [
            'What is Power BI and what are its main components?',
            'What is the difference between Power Query and DAX?',
            'What is the difference between a measure and a calculated column?',
            'How do relationships and cardinality work in Power BI?',
            'What is a star schema and why is it useful?',
            'What is filter context in DAX?',
            'Which common DAX functions have you used?',
            'How does data refresh work in Power BI?',
            'What are the key principles of designing an effective Power BI dashboard?',
            'How would you improve the performance of a slow Power BI report?',
            'What is the difference between Import mode and DirectQuery in Power BI?',
            'What is row context in DAX and how is it different from filter context?',
            'What does the CALCULATE function do in DAX?',
            'What is Row-Level Security in Power BI?',
            'What are drill-down and drill-through in Power BI?',
            'How do slicers and filters affect report interaction?',
            'How would you create a year-over-year growth measure in Power BI?',
            'How would you handle many-to-many relationships in a Power BI data model?',
            'A dashboard is showing an incorrect total. How would you investigate the issue?',
            'How would you design a management dashboard that highlights KPIs, trends and exceptions clearly?',
        ],
        'Excel': [
            'What is the difference between VLOOKUP and XLOOKUP?',
            'How do INDEX and MATCH work together?',
            'What is a PivotTable and when would you use it?',
            'How do IF and IFS functions work?',
            'What are SUMIFS and COUNTIFS used for?',
            'How would you clean a dataset in Excel?',
            'What is conditional formatting and when is it useful?',
            'How would you build a simple dashboard in Excel?',
            'What is the difference between relative and absolute cell references?',
            'What is Power Query in Excel and why is it useful?',
            'What is the difference between COUNT, COUNTA and COUNTIF?',
            'How do IFERROR and error-handling functions help in Excel?',
            'What are Excel Tables and what advantages do they provide?',
            'How do text functions such as LEFT, RIGHT, MID and TEXTSPLIT help in data cleaning?',
            'How do date and time functions help in business analysis?',
            'What are named ranges and when are they useful?',
            'How would you identify and remove duplicate records in Excel?',
            'How would you combine data from multiple files using Power Query?',
            'A large Excel workbook is running slowly. How would you improve its performance?',
            'How would you build an automated monthly reporting workflow using Excel and Power Query?',
        ],
        'Artificial Intelligence': [
            'What is Artificial Intelligence?',
            'What is the difference between AI, Machine Learning and Deep Learning?',
            'What is the difference between narrow AI and general AI?',
            'What are some common real-world applications of AI?',
            'What is the difference between training and inference in an AI system?',
            'How are NLP and Computer Vision used in AI?',
            'What are bias and responsible AI?',
            'What are some important limitations of AI systems?',
            'How can the performance of an AI solution be evaluated?',
            'What are the main stages of an AI project from business problem to deployment?',
            'What is the difference between rule-based systems and learning-based AI systems?',
            'What are supervised, unsupervised and reinforcement learning in the context of AI?',
            'What are features, labels and predictions in an AI system?',
            'What is explainable AI and why can it be important?',
            'What is human-in-the-loop AI?',
            'What is model drift and why should deployed AI systems be monitored?',
            'How would you identify whether a business problem actually requires AI?',
            'What factors would you consider before deploying an AI solution in a real business environment?',
            'How would you reduce the risk of incorrect AI predictions affecting users?',
            'How would you explain the value and limitations of an AI solution to a non-technical manager?',
        ],
        'Deep Learning': [
            'What is Deep Learning and how is it different from traditional Machine Learning?',
            'What is an artificial neural network?',
            'What are neurons, weights and biases in a neural network?',
            'What are forward propagation and backpropagation?',
            'What are activation functions and why are they needed?',
            'What are a loss function and an optimizer?',
            'What is a Convolutional Neural Network?',
            'What are RNNs and LSTMs?',
            'What is dropout and how does it help reduce overfitting?',
            'What is transfer learning?',
            'What is the difference between an epoch, batch and iteration in Deep Learning?',
            'What are vanishing and exploding gradients?',
            'What is batch normalization and why is it used?',
            'What is the difference between SGD and Adam optimizers?',
            'What is early stopping and how does it help training?',
            'What is data augmentation and when is it useful?',
            'What is the difference between CNNs, RNNs and Transformers?',
            'How would you choose an activation function for different layers of a neural network?',
            'A neural network performs very well on training data but poorly on validation data. What would you try?',
            'When would you choose transfer learning instead of training a Deep Learning model from scratch?',
        ],
        'NLP': [
            'What is Natural Language Processing and where is it used?',
            'What is tokenization?',
            'What are stop-word removal, stemming and lemmatization?',
            'What are Bag of Words and TF-IDF?',
            'What are word embeddings?',
            'What is sentiment analysis?',
            'What is Named Entity Recognition?',
            'How are transformers used in NLP?',
            'What is the attention mechanism?',
            'What are some common challenges when building NLP systems?',
            'What is the difference between stemming and lemmatization?',
            'What are n-grams and when are they useful?',
            'What is cosine similarity and how can it be used with text vectors?',
            'What is the difference between static word embeddings and contextual embeddings?',
            'What is sequence-to-sequence learning?',
            'What are encoder and decoder architectures in NLP?',
            'How does self-attention help a transformer understand relationships between words?',
            'How would you evaluate a text classification model?',
            'How would you handle multilingual or mixed-language text in an NLP application?',
            'If an NLP model misunderstands domain-specific terminology, how would you improve the system?',
        ],
        'Generative AI': [
            'What is Generative AI?',
            'What is a Large Language Model?',
            'What is a transformer model at a high level?',
            'What is prompt engineering?',
            'What are tokens and a context window?',
            'What does temperature control in a generative AI model?',
            'What is an AI hallucination and how can it be reduced?',
            'What is Retrieval-Augmented Generation?',
            'What is the difference between prompting, RAG and fine-tuning?',
            'What are embeddings and vector databases, and how are they used in Generative AI?',
            'What is semantic search and how is it different from keyword search?',
            'What is cosine similarity in an embedding-based retrieval system?',
            'What is a vector store and what role does it play in RAG?',
            'What are system prompts, user prompts and model responses?',
            'What are zero-shot and few-shot prompting?',
            'What is fine-tuning and when would you consider it instead of RAG?',
            'What are AI agents and how are they different from a normal LLM call?',
            'What roles can LangChain and LangGraph play in a Generative AI application?',
            'How would you evaluate the quality and reliability of a RAG-based application?',
            'How would you add guardrails to prevent malformed or unreliable AI output from affecting an application?',
        ],
        'Project Questions': [
            'Explain the problem statement of one of your projects.',
            'What was the objective and business value of your project?',
            'What data did you use and where did it come from?',
            'How did you clean and preprocess the data?',
            'What technologies, algorithms or models did you use and why?',
            'What was the biggest challenge in your project and how did you solve it?',
            'How did you evaluate the performance of your solution?',
            'How did you convert your model or analysis into a usable application?',
            'What result or practical value did the project provide?',
            'What would you improve if you developed the project again?',
            'Why did you choose this project instead of a simpler problem?',
            'Which part of your project was personally implemented by you?',
            'How did you decide which features were important for your solution?',
            'What alternatives did you consider before selecting your final approach?',
            'What limitations does your current project have?',
            'How would your application handle new or unseen real-world data?',
            'How would you monitor the project after deploying it for real users?',
            'How did you handle errors, invalid inputs or AI failures in your application?',
            'How would you scale your project if many users started using it at the same time?',
            'If an interviewer gave you one more month to improve this project, what would you add and why?',
        ]
    }

    topic_options = list(techprep_banks.keys()) + ["Custom"]

    # --------------------------------------------------------
    # TOPIC
    # --------------------------------------------------------

    selected_topic = st.selectbox(
        "📚 Choose Technical Topic",
        topic_options,
        key="techprep_topic"
    )

    if selected_topic == "Custom":

        custom_topic = st.text_input(
            "Enter your topic",
            key="techprep_custom_topic",
            placeholder="Example: Time Series Forecasting"
        )

        selected_question = st.text_area(
            "Enter your interview question",
            key="techprep_custom_question",
            height=100,
            placeholder="Type the technical question you want to practise..."
        )

        effective_topic = custom_topic.strip() or "Custom"

    else:

        effective_topic = selected_topic

        questions = techprep_banks[selected_topic]
        total_available_questions = len(questions)

        question_count_options = [5, 10, 15, 20, "All Important"]

        selected_question_count = st.selectbox(
            "Number of Important Questions",
            question_count_options,
            index=3,
            key="techprep_question_count",
            help=(
                "Choose how many important questions you want to practise. "
                "If a topic contains fewer questions, FluentPath automatically "
                "uses all available questions."
            )
        )

        if selected_question_count == "All Important":
            active_question_count = total_available_questions
        else:
            active_question_count = min(
                int(selected_question_count),
                total_available_questions
            )

        active_questions = questions[:active_question_count]

        question_number = st.selectbox(
            "Choose Question",
            range(1, active_question_count + 1),
            format_func=lambda x: f"Question {x}",
            key="techprep_question_number"
        )

        selected_question = active_questions[question_number - 1]

        st.markdown("### Interview Question")
        st.info(selected_question)

        st.caption(
            f"Question {question_number} of {active_question_count} • "
            f"{selected_topic} • {total_available_questions} important questions available"
        )

    # --------------------------------------------------------
    # SESSION CACHE
    # --------------------------------------------------------

    current_key = f"{effective_topic}::{selected_question}".strip()

    if st.session_state.get("techprep_current_key") != current_key:
        st.session_state["techprep_current_key"] = current_key
        st.session_state["techprep_learning_report"] = ""
        st.session_state["techprep_practice_report"] = ""
        st.session_state["techprep_practice_score"] = None
        st.session_state["techprep_concept_report"] = ""

    # --------------------------------------------------------
    # LEARN
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 🧠 Learn the Answer")

    st.caption(
        "Generate the explanation only when you need it. "
        "This keeps TechPrep focused and easy to revise."
    )

    if st.button(
        "✨ Explain This Question",
        use_container_width=True,
        key="techprep_generate_learning"
    ):

        if not selected_question.strip():
            st.warning("Enter a question first.")

        else:
            learning_system = """
You are TechPrep, a technical interview learning coach.

The learner is preparing for technical interviews.

For the given topic and question, teach the concept accurately and
practically.

Use exactly these sections:

## தமிழ் விளக்கம்
Explain the concept in simple, clear Tamil.
Keep standard technical terms such as Python, Machine Learning, SQL,
API, model, dataset and algorithm in English when appropriate.

## Simple Interview English
Give a short, natural answer that a learner can comfortably speak in
an interview. Use simple English. Do not make it sound memorized.

## Professional Interview Answer
Give a stronger professional answer with correct technical points.
Keep it interview-friendly and reasonably concise.

## Key Points to Remember
Give 4 to 6 short revision points.

Important:
- Do not invent the learner's personal experience.
- For project questions, give an answer structure/template unless the
  learner has supplied real project details.
- Keep technical information accurate.
- Use the Retrieved Knowledge Context supplied in the user prompt as
  grounding when it is relevant.
- Do not force irrelevant retrieved information into the answer.
- If the retrieved context is incomplete, carefully use your technical
  knowledge to complete the explanation.
- Do not claim that retrieved context contains information that it does not.
- Never repeat the same sentence, phrase, bullet point, or explanation.
- Keep each section concise and remove duplicated ideas.
- Tamil explanation must use natural Tamil with standard technical terms
  such as RAG, embedding, vector database, retrieval, prompt and LLM in English.
- Simple Interview English and Professional Interview Answer must be written
  in English only.
- Professional Interview Answer should normally be one concise paragraph,
  followed by useful points only when necessary.
- Before returning the final response, check for accidental repetition and
  rewrite any repeated content.
- Do not add fake tools, results, percentages or achievements.
"""

            # RAG: retrieve relevant technical knowledge before
            # sending the question to the language model.
            learning_user = build_rag_user_prompt(
                effective_topic,
                selected_question
            )

            with st.spinner("TechPrep AI is preparing your answer..."):
                try:
                    # Primary:
                    # Agentic TechPrep
                    try:
                        agent_result = run_techprep_agent(
                            topic=effective_topic,
                            question=selected_question,
                            user_id=st.session_state.get("user_id")
                        )

                        learning_report = agent_result.get(
                            "final_answer",
                            ""
                        )

                        if not learning_report:
                            raise RuntimeError(
                                "Agentic TechPrep returned "
                                "an empty answer."
                            )

                        st.session_state[
                            "techprep_agent_meta"
                        ] = {
                            "intent": agent_result.get(
                                "intent",
                                "INTERVIEW"
                            ),
                            "action": agent_result.get(
                                "agent_action",
                                "Generate Interview Answer"
                            ),
                            "difficulty": agent_result.get(
                                "recommended_difficulty",
                                "Beginner"
                            ),
                            "quality": agent_result.get(
                                "quality_status",
                                "PASS"
                            ),
                        }

                    except Exception:
                        # Fallback 1:
                        # Verified LangGraph workflow.
                        try:
                            graph_result = generate_techprep_graph_answer(
                                effective_topic,
                                selected_question
                            )

                            learning_report = graph_result.get(
                                "final_answer",
                                graph_result.get("answer", "")
                            )

                            if not learning_report:
                                raise RuntimeError(
                                    "LangGraph returned "
                                    "an empty answer."
                                )

                        except Exception:
                            # Fallback 2:
                            # Verified LangChain pipeline.
                            try:
                                learning_report = generate_techprep_answer(
                                    effective_topic,
                                    selected_question
                                )

                            except Exception:
                                # Fallback 3:
                                # Verified direct RAG + ask_ai.
                                learning_report = ask_ai(
                                    learning_system,
                                    learning_user
                                )

                    st.session_state[
                        "techprep_learning_report"
                    ] = learning_report

                except Exception as e:
                    st.error(
                        f"AI response could not be generated: {e}"
                    )

    learning_report = st.session_state.get(
        "techprep_learning_report",
        ""
    )

    if learning_report:
        st.markdown(learning_report)

    # --------------------------------------------------------
    # I KNOW THE CONCEPT
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 💡 I Know the Concept")

    st.caption(
        "Explain what you know in Tamil, English, or mixed language. "
        "TechPrep will help you convert your idea into an interview answer."
    )

    concept_reset = st.session_state.get(
        "techprep_concept_reset",
        0
    )

    concept_text = st.text_area(
        "Explain the concept in your own words",
        height=170,
        key=f"techprep_concept_text_{concept_reset}",
        placeholder=(
            "Tamil / English / mixed language — explain what you understand..."
        )
    )

    c1, c2 = st.columns(2)

    with c1:
        concept_submit = st.button(
            "🧠 Structure My Concept",
            use_container_width=True,
            key="techprep_structure_concept"
        )

    with c2:
        if st.button(
            "🗑️ Clear Concept",
            use_container_width=True,
            key="techprep_clear_concept"
        ):
            st.session_state["techprep_concept_reset"] = (
                concept_reset + 1
            )
            st.session_state["techprep_concept_report"] = ""
            st.rerun()

    if concept_submit:

        if not concept_text.strip():
            st.warning(
                "Explain what you know first. Tamil or simple English is fine."
            )

        elif not selected_question.strip():
            st.warning("Choose or enter an interview question first.")

        else:
            concept_system = """
You are a supportive technical interview coach.

The learner may explain a technical concept in Tamil, English,
broken English, or a mixture of languages.

Do not reject Tamil.

First understand the learner's intended technical idea.
Do not invent experience or achievements.

Use exactly these sections:

## What You Already Understand
Briefly identify the correct ideas in the learner's explanation.

## Important Correction
Correct technical misunderstandings clearly and respectfully.
If there is no major technical mistake, say that the core idea is correct.

## Simple Interview English
Convert the learner's idea into natural, simple spoken English.

## Professional Version
Give a technically accurate and professional interview version.

## Missing Point
Mention only the most useful missing technical points.

Keep the response practical and concise.
"""

            concept_user = f"""
Topic:
{effective_topic}

Interview Question:
{selected_question}

Learner's Own Explanation:
{concept_text}
"""

            with st.spinner(
                "TechPrep AI is structuring your concept..."
            ):
                try:
                    concept_report = ask_ai(
                        concept_system,
                        concept_user
                    )

                    st.session_state[
                        "techprep_concept_report"
                    ] = concept_report

                except Exception as e:
                    st.error(
                        f"AI response could not be generated: {e}"
                    )

    concept_report = st.session_state.get(
        "techprep_concept_report",
        ""
    )

    if concept_report:
        st.markdown(concept_report)

    # --------------------------------------------------------
    # PRACTICE
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 🎤 Practice This Answer")

    st.caption(
        "Answer in English as you would in an interview. "
        "Your answer does not need to be perfect."
    )

    practice_reset = st.session_state.get(
        "techprep_practice_reset",
        0
    )

    practice_answer = st.text_area(
        "Your Interview Answer",
        height=220,
        key=f"techprep_practice_answer_{practice_reset}",
        placeholder=(
            "Type your answer in your own words..."
        )
    )

    p1, p2 = st.columns(2)

    with p1:
        analyze_practice = st.button(
            "🤖 Analyze My Answer",
            use_container_width=True,
            key="techprep_analyze_practice"
        )

    with p2:
        if st.button(
            "🔄 Try Again",
            use_container_width=True,
            key="techprep_retry"
        ):
            st.session_state["techprep_practice_reset"] = (
                practice_reset + 1
            )
            st.session_state["techprep_practice_report"] = ""
            st.session_state["techprep_practice_score"] = None
            st.rerun()

    if analyze_practice:

        if not practice_answer.strip():
            st.warning(
                "Write a short interview answer first."
            )

        elif not selected_question.strip():
            st.warning(
                "Choose or enter an interview question first."
            )

        else:
            practice_system = """
You are TechPrep, a technical interview evaluator.

Evaluate only the learner's written answer to the given technical
interview question.

Score these five areas:
1. Concept Coverage - 20
2. Technical Accuracy - 20
3. English Clarity - 20
4. Answer Structure - 20
5. Interview Relevance - 20

Total = 100.

Do not evaluate pronunciation, accent, voice quality or speaking speed.

Use exactly this format:

## TechPrep Score
Total Score: <0-100>/100

- Concept Coverage: <0-20>/20
- Technical Accuracy: <0-20>/20
- English Clarity: <0-20>/20
- Answer Structure: <0-20>/20
- Interview Relevance: <0-20>/20

## What Was Good
Give the strongest points briefly.

## Technical Correction
Correct any inaccurate technical statement.
If there is no major technical error, state that clearly.

## Missing Points
Give only useful missing points.

## Better Answer
Rewrite the learner's answer in clear, natural interview English.
Preserve the learner's intended meaning and do not invent personal
experience.

## Next Practice
Give one short action for the next attempt.
"""

            practice_user = f"""
Topic:
{effective_topic}

Interview Question:
{selected_question}

Learner Answer:
{practice_answer}
"""

            with st.spinner(
                "TechPrep AI is analyzing your answer..."
            ):
                try:
                    practice_report = ask_ai(
                        practice_system,
                        practice_user
                    )

                    import re

                    # ====================================
                    # TECHPREP FLEXIBLE EXPLICIT SCALE PARSER V2
                    # ====================================
                    #
                    # Accept common Markdown variations such as:
                    #
                    # Total Score: 85/100
                    # **Total Score:** 85/100
                    # Total Score **:** 85/100
                    # Total Score - 85/100
                    # Overall Score: 85/100
                    # Score: 85/100
                    # Total Score: 85%
                    #
                    # IMPORTANT:
                    # Only explicit /100 or % scores are accepted.
                    # Plain low scores such as "8" remain blocked.
                    # ====================================

                    score_patterns = [
                        (
                            r"(?:Total|Overall)\s*Score"
                            r"\s*\*{0,2}\s*[:=-]\s*\*{0,2}\s*"
                            r"(\d{1,3})\s*/\s*100"
                        ),
                        (
                            r"(?:Total|Overall)\s*Score"
                            r"\s*\*{0,2}\s*[:=-]\s*\*{0,2}\s*"
                            r"(\d{1,3})\s*%"
                        ),
                        (
                            r"(?<![A-Za-z])Score"
                            r"\s*\*{0,2}\s*[:=-]\s*\*{0,2}\s*"
                            r"(\d{1,3})\s*/\s*100"
                        ),
                        (
                            r"(?<![A-Za-z])Score"
                            r"\s*\*{0,2}\s*[:=-]\s*\*{0,2}\s*"
                            r"(\d{1,3})\s*%"
                        )
                    ]

                    # ====================================
                    # TECHPREP SCORE SCALE GUARDRAIL V1
                    # ====================================
                    #
                    # Preserve explicit /100 and percentage
                    # score contracts from the AI response.
                    #
                    # Examples:
                    #     8/100  -> valid 8
                    #     82/100 -> valid 82
                    #     82%    -> valid 82
                    #
                    # A plain low score such as "8" remains
                    # ambiguous and must not become 80.
                    # ====================================

                    techprep_score = None
                    _raw_techprep_score = None

                    for pattern in score_patterns:
                        match = re.search(
                            pattern,
                            practice_report,
                            re.IGNORECASE
                        )

                        if match:
                            _score_number = match.group(1)
                            _matched_text = match.group(0)

                            if "/100" in _matched_text.replace(" ", ""):
                                _raw_techprep_score = (
                                    f"{_score_number}/100"
                                )

                            elif "%" in _matched_text:
                                _raw_techprep_score = (
                                    f"{_score_number}%"
                                )

                            else:
                                _raw_techprep_score = (
                                    _score_number
                                )

                            break

                    if _raw_techprep_score is None:
                        # Keep the guardrail strict.
                        # Do not invent or convert a missing score.
                        #
                        # Store a short diagnostic preview locally
                        # in session state for controlled debugging.
                        st.session_state[
                            "techprep_guardrail_diagnostic"
                        ] = practice_report[:500]

                        raise ValueError(
                            "TechPrep AI response did not return "
                            "an explicit /100 or percentage score."
                        )

                    _techprep_score_guard = (
                        validate_score_contract(
                            _raw_techprep_score
                        )
                    )

                    if not _techprep_score_guard.get("valid"):
                        _guard_status = (
                            _techprep_score_guard.get(
                                "status",
                                "UNKNOWN"
                            )
                        )

                        raise ValueError(
                            "TechPrep returned an ambiguous "
                            "or invalid score. "
                            f"Guardrail status: {_guard_status}"
                        )

                    techprep_score = int(
                        round(
                            float(
                                _techprep_score_guard["score"]
                            )
                        )
                    )

                    st.session_state[
                        "techprep_practice_report"
                    ] = practice_report

                    st.session_state[
                        "techprep_practice_score"
                    ] = techprep_score

                    try:
                        save_activity(
                            activity_type="TechPrep",
                            activity=(
                                f"{effective_topic} - "
                                f"{selected_question[:80]}"
                            ),
                            score=techprep_score,
                            review_data={
                                "title": (
                                    f"TechPrep - {effective_topic}"
                                ),
                                "question": selected_question,
                                "answer": practice_answer,
                                "feedback": practice_report,
                                "score": techprep_score
                            }
                        )
                    except TypeError:
                        # Compatibility with older save_activity signature.
                        save_activity(
                            activity_type="TechPrep",
                            activity=(
                                f"{effective_topic} - "
                                f"{selected_question[:80]}"
                            ),
                            score=techprep_score
                        )

                except Exception as e:
                    st.error(
                        f"AI analysis could not be completed: {e}"
                    )

    practice_report = st.session_state.get(
        "techprep_practice_report",
        ""
    )

    practice_score = st.session_state.get(
        "techprep_practice_score"
    )

    if practice_report:

        st.markdown("### 📊 TechPrep Feedback")

        if practice_score is not None:
            st.metric(
                "Technical Interview Practice Score",
                f"{practice_score}%"
            )

        st.markdown(practice_report)

    # --------------------------------------------------------
    # REVISION LIST
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 📌 My Revision List")

    st.caption(
        "Your scored TechPrep practice is saved to your learning history."
    )

    db = SessionLocal()

    try:
        techprep_logs = (
            db.query(ActivityLog)
            .filter(
                ActivityLog.user_id
                == st.session_state.user_id,
                ActivityLog.activity_type
                == "TechPrep"
            )
            .order_by(
                ActivityLog.created_at.desc()
            )
            .limit(10)
            .all()
        )
    finally:
        db.close()

    if techprep_logs:

        revision_scores = [
            log.score
            for log in techprep_logs
            if log.score is not None
        ]

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Practised",
            len(techprep_logs)
        )

        if revision_scores:
            r2.metric(
                "Average",
                f"{round(sum(revision_scores) / len(revision_scores))}%"
            )
            r3.metric(
                "Best",
                f"{max(revision_scores)}%"
            )
        else:
            r2.metric("Average", "—")
            r3.metric("Best", "—")

        st.markdown("### Recent TechPrep Practice")

        for log in techprep_logs[:5]:
            score_text = (
                f"{int(log.score)}%"
                if log.score is not None
                else "—"
            )

            st.write(
                f"• **{log.activity}** — {score_text}"
            )

        st.info(
            "Open 📜 History to review saved TechPrep activity."
        )

    else:
        st.info(
            "Complete your first TechPrep answer to start your revision list."
        )




# ============================================================
# SMARTSPEAK - THINK YOUR WAY. SPEAK WITH CONFIDENCE.
# ============================================================


# ============================================================
# TOPIC 10 - AI ARCHITECTURE V1
# ============================================================

elif page == "🧠 AI Architecture":

    friendly_coach_message("🧠 AI Architecture")

    st.title("🧠 AI Architecture")

    st.caption(
        "A clear view of how FluentPath combines AI, retrieval, "
        "learning intelligence, workflow orchestration and persistent data."
    )

    st.info(
        "FluentPath uses existing transformer-based language models through "
        "Gemini, OpenRouter and Groq, in that fallback order. "
        "It does not train a custom Transformer model."
    )

    # --------------------------------------------------------
    # HIGH-LEVEL FLOW
    # --------------------------------------------------------

    st.markdown("## 🔄 End-to-End Architecture")

    st.code(
        """User
  ↓
Streamlit Interface
  ↓
Input Processing / Guardrails
  ↓
Learning Memory
  ↓
Adaptive Learning Engine
  ↓
Agentic TechPrep
  ↓
RAG
  ↓
Embeddings + Vector Retrieval
  ↓
LangGraph Workflow
  ↓
LangChain Pipeline
  ↓
Gemini → OpenRouter → Groq (language feedback)
  ↓
Output Validation
  ↓
Learner Feedback / Score / Recommendation
  ↓
SQLAlchemy
  ↓
SQLite (local) or PostgreSQL / Neon (configured cloud)""",
        language="text"
    )

    # --------------------------------------------------------
    # ARCHITECTURE EXPLANATION
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 🧩 Architecture Layers")

    with st.expander("1️⃣ Streamlit — User Interface", expanded=True):
        st.write(
            "Streamlit provides the interactive application interface for "
            "Dashboard, Learning, Practice, Tests, Daily Assignment, "
            "Speaking, Interviews, TechPrep, SmartSpeak, Progress and History."
        )

    with st.expander("2️⃣ Guardrails — Reliability Layer"):
        st.write(
            "Guardrails validate important AI outputs before they are used "
            "by the application. Score validation helps prevent malformed "
            "or ambiguous AI scores from becoming learner data."
        )

    with st.expander("3️⃣ Learning Memory — Persistent Learning State"):
        st.write(
            "Learning Memory reads the learner's saved activity history "
            "and builds a practical learning state from previous scored "
            "activities."
        )
        st.caption(
            "Current implementation: deterministic database-history aggregation."
        )

    with st.expander("4️⃣ Adaptive Learning Engine"):
        st.write(
            "The Adaptive Learning Engine analyses previous TechPrep "
            "performance to identify learning level, trend and recommended "
            "practice action."
        )
        st.caption(
            "Current implementation: deterministic rule-based personalization, "
            "not a separately trained machine-learning model."
        )

    with st.expander("5️⃣ Agentic TechPrep"):
        st.write(
            "Agentic TechPrep coordinates learner intent, adaptive context, "
            "retrieval and answer-generation workflows."
        )

    with st.expander("6️⃣ RAG — Retrieval-Augmented Generation"):
        st.write(
            "RAG retrieves relevant TechPrep knowledge before generation "
            "and supplies that context to the language model."
        )

    with st.expander("7️⃣ Embeddings & Vector Retrieval"):
        st.write(
            "FluentPath can use OpenRouter embeddings to represent text as "
            "vectors. Cosine similarity is used for semantic retrieval, "
            "with a local lexical fallback available."
        )

    with st.expander("8️⃣ LangGraph — Workflow Orchestration"):
        st.write(
            "LangGraph manages the stateful TechPrep workflow."
        )
        st.code(
            "Retrieve → Generate → Evaluate → Improve / Finalize",
            language="text"
        )
        st.write(
            "Conditional routing allows the workflow to decide whether "
            "an answer should be improved or finalized."
        )

    with st.expander("9️⃣ LangChain — LLM Orchestration"):
        st.write(
            "LangChain connects prompt templates, retrieved context, "
            "the language model and output parsing into a reusable pipeline."
        )

    with st.expander("🔟 Transformer-Based LLM"):
        st.write(
            "FluentPath's direct coaching requests try Gemini, then "
            "OpenRouter, then Groq. TechPrep's specialized LangChain, "
            "LangGraph and embeddings integrations use OpenRouter."
        )
        st.warning(
            "FluentPath does not train its own Transformer or Deep Learning "
            "model. It integrates existing transformer-based LLMs."
        )

    with st.expander("1️⃣1️⃣ Output Validation"):
        st.write(
            "Important AI-generated outputs are validated before they are "
            "accepted by application workflows. Invalid or ambiguous score "
            "formats can be rejected instead of silently becoming incorrect data."
        )

    with st.expander("1️⃣2️⃣ SQLAlchemy — Persistent Data"):
        st.write(
            "SQLAlchemy connects the application to the persistent database. "
            "The local application uses SQLite. A configured cloud deployment "
            "can use PostgreSQL / Neon for users, activities, scores and history."
        )

    # --------------------------------------------------------
    # TECHNOLOGY CLASSIFICATION
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## ⚙️ Technology Responsibilities")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
### AI / Generative Layer

**Gemini, OpenRouter and Groq**
- Direct AI coaching uses Gemini first, then OpenRouter and Groq as fallbacks
- OpenRouter also supports specialized TechPrep integrations

**RAG**
- Retrieves useful knowledge before generation

**Embeddings**
- Semantic vector representation

**LangChain**
- Prompt, model and output orchestration

**LangGraph**
- Stateful conditional workflow

**Agentic TechPrep**
- Coordinates multiple AI workflow stages
            """
        )

    with col2:
        st.markdown(
            """
### Application / Intelligence Layer

**Learning Memory**
- Uses saved learner history

**Adaptive Learning**
- Rule-based personalization

**Guardrails**
- AI output and score validation

**Streamlit**
- Application interface

**SQLAlchemy**
- Database interaction layer

**SQLite / Neon PostgreSQL**
- SQLite locally; PostgreSQL when configured in the cloud
            """
        )

    # --------------------------------------------------------
    # AI VS DETERMINISTIC
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 🤖 AI vs Deterministic Components")

    st.markdown(
        """
| Component | Implementation |
|---|---|
| AI feedback & generation | Gemini → OpenRouter → Groq fallback |
| RAG retrieval | Embeddings + cosine similarity with lexical fallback |
| LangChain | Prompt / model / output pipeline |
| LangGraph | Stateful conditional workflow |
| Agentic TechPrep | Intent, retrieval and generation orchestration |
| Adaptive Learning | Deterministic rule-based analysis |
| Learning Memory | Deterministic database-history aggregation |
| Interview Intelligence V2 | Deterministic structural intelligence using existing AI score |
| Score Guardrails | Deterministic validation |
| Face Detection | OpenCV Haar Cascade |
| Speech-to-Text | Google Speech Recognition through SpeechRecognition |
| Persistent Storage | SQLAlchemy + local SQLite or configured PostgreSQL / Neon |
        """
    )

    # --------------------------------------------------------
    # TRANSFORMER EXPLANATION
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 🧠 Where Does the Transformer Fit?")

    st.success(
        "The Transformer is inside the Large Language Model layer. "
        "FluentPath sends prompts and retrieved context to transformer-based "
        "LLMs through Gemini, OpenRouter or Groq and then validates the returned output."
    )

    st.markdown(
        """
**RAG** decides what relevant context should be supplied.

**Embeddings** help retrieve semantically relevant knowledge.

**LangChain** structures the model interaction.

**LangGraph** controls the AI workflow.

**Transformer-based LLM** performs language understanding and generation.

**Guardrails** validate important outputs.

**Learning Memory + Adaptive Learning** personalize the learner experience.
        """
    )

    # --------------------------------------------------------
    # INTERVIEW EXPLANATION
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 🎤 Interview-Ready Explanation")

    st.write(
        "FluentPath is a multi-user AI English coaching application built "
        "with Streamlit and SQLAlchemy (SQLite locally, PostgreSQL when configured). "
        "For advanced technical coaching, "
        "it uses Retrieval-Augmented Generation with embeddings and vector "
        "similarity to retrieve relevant context. LangChain manages prompt "
        "and model orchestration, while LangGraph controls the stateful "
        "retrieve, generate, evaluate and improve workflow. Transformer-based "
        "language models are accessed through Gemini, OpenRouter or Groq for generation and "
        "AI evaluation. The application also uses persistent learning memory, "
        "a rule-based adaptive learning engine and deterministic guardrails "
        "to improve personalization and output reliability."
    )

    st.caption(
        "Technical accuracy: FluentPath integrates existing transformer-based "
        "LLMs. It does not train a custom Transformer model."
    )


elif page == "💬 SmartSpeak":

    friendly_coach_message("💬 SmartSpeak")

    st.title("💬 SmartSpeak")
    st.markdown("### Think Your Way. Speak with Confidence.")

    st.info(
        "Write what you want to say in Tamil, English, broken English, "
        "or mixed language. SmartSpeak will keep your meaning and help "
        "you express it naturally in English."
    )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    situations = [
        "Daily Conversation",
        "Workplace",
        "Customer / Business",
        "Career / Interview",
        "Presentation / Public Speaking",
        "Custom"
    ]

    tones = [
        "Simple",
        "Natural",
        "Professional",
        "Polite",
        "Confident",
        "Friendly",
        "Formal",
        "Concise"
    ]

    c1, c2 = st.columns(2)

    with c1:
        situation = st.selectbox(
            "🎯 Situation",
            situations,
            key="smartspeak_situation"
        )

    with c2:
        tone = st.selectbox(
            "🎭 Preferred Tone",
            tones,
            key="smartspeak_tone"
        )

    custom_context = ""

    if situation == "Custom":
        custom_context = st.text_input(
            "Describe the situation",
            key="smartspeak_custom_context",
            placeholder=(
                "Example: I need to speak to my manager about a delay."
            )
        )

    effective_situation = (
        custom_context.strip()
        if situation == "Custom" and custom_context.strip()
        else situation
    )

    # --------------------------------------------------------
    # YOUR THOUGHT
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 💭 Your Thought")

    st.caption(
        "Tamil / English / mixed language / broken English — all are welcome."
    )

    thought_reset = st.session_state.get(
        "smartspeak_thought_reset",
        0
    )

    thought = st.text_area(
        "What do you want to say?",
        height=180,
        key=f"smartspeak_thought_{thought_reset}",
        placeholder=(
            "Example: Naalaiku meeting konjam late ah start panna mudiyuma? "
            "I have another work in morning."
        )
    )

    b1, b2 = st.columns(2)

    with b1:
        convert_thought = st.button(
            "✨ Make My English Better",
            use_container_width=True,
            key="smartspeak_convert"
        )

    with b2:
        clear_thought = st.button(
            "🗑️ Clear",
            use_container_width=True,
            key="smartspeak_clear"
        )

    if clear_thought:
        st.session_state["smartspeak_thought_reset"] = (
            thought_reset + 1
        )
        st.session_state["smartspeak_report"] = ""
        st.session_state["smartspeak_original_thought"] = ""
        st.session_state["smartspeak_practice_report"] = ""
        st.session_state["smartspeak_practice_score"] = None
        st.rerun()

    if convert_thought:

        if not thought.strip():
            st.warning(
                "Write your thought first. Tamil, English, or mixed language is fine."
            )

        else:
            smart_system = """
You are SmartSpeak, a practical English communication coach.

The learner may write in:
- Tamil
- English
- Tamil + English
- broken English
- incomplete sentences

Understand the learner's intended meaning first.

Do NOT reject Tamil or mixed-language input.
Do NOT change the learner's intended meaning.
Do NOT invent facts, names, experience, achievements or commitments.

The learner wants to use the sentence in a real-life situation.

Use exactly these sections:

## Your Meaning
In one short sentence, state what the learner is trying to communicate.
This section may use simple Tamil + English if the original thought is Tamil
or mixed language.

## Correct English
Correct the grammar while staying close to the learner's original wording.

## Natural English
Give the version a fluent English speaker could naturally use in this
situation.

## Professional Version
Give a polished workplace-appropriate version.
Keep the same meaning.

## Best Version for This Situation
Give ONE final recommended sentence or short response matching the requested
tone and situation.

## Why This Is Better
Give 2 to 4 short learning points explaining the important improvements.
Focus on grammar, wording, clarity, politeness or natural expression.

Important:
Correct English, Natural English, Professional Version and Best Version
must be written in ENGLISH.
Do not translate those sections into Tamil.
Keep the response practical and concise.
"""

            smart_user = f"""
Situation:
{effective_situation}

Requested Tone:
{tone}

Learner's Thought:
{thought}
"""

            with st.spinner(
                "SmartSpeak is improving your message..."
            ):
                try:
                    smart_report = ask_ai(
                        smart_system,
                        smart_user
                    )

                    st.session_state[
                        "smartspeak_report"
                    ] = smart_report

                    st.session_state[
                        "smartspeak_original_thought"
                    ] = thought

                    st.session_state[
                        "smartspeak_last_situation"
                    ] = effective_situation

                    st.session_state[
                        "smartspeak_last_tone"
                    ] = tone

                except Exception as e:
                    st.error(
                        f"SmartSpeak could not generate the response: {e}"
                    )

    smart_report = st.session_state.get(
        "smartspeak_report",
        ""
    )

    if smart_report:
        st.markdown("---")
        st.markdown("## 🌟 SmartSpeak Result")
        st.markdown(smart_report)

    # --------------------------------------------------------
    # PRACTICE
    # --------------------------------------------------------

    if smart_report:

        st.markdown("---")
        st.markdown("## 🎤 Practice Your Version")

        st.caption(
            "Now say or type the idea again in your own English. "
            "Do not worry about making it exactly the same as the AI version."
        )

        practice_reset = st.session_state.get(
            "smartspeak_practice_reset",
            0
        )

        practice_text = st.text_area(
            "Your English Practice",
            height=160,
            key=f"smartspeak_practice_{practice_reset}",
            placeholder=(
                "Write the improved sentence again in your own words..."
            )
        )

        p1, p2 = st.columns(2)

        with p1:
            analyze_practice = st.button(
                "🤖 Check My English",
                use_container_width=True,
                key="smartspeak_check_practice"
            )

        with p2:
            retry_practice = st.button(
                "🔄 Practice Again",
                use_container_width=True,
                key="smartspeak_retry"
            )

        if retry_practice:
            st.session_state["smartspeak_practice_reset"] = (
                practice_reset + 1
            )
            st.session_state["smartspeak_practice_report"] = ""
            st.session_state["smartspeak_practice_score"] = None
            st.rerun()

        if analyze_practice:

            if not practice_text.strip():
                st.warning(
                    "Write your English practice sentence first."
                )

            else:
                practice_system = """
You are SmartSpeak, a supportive English communication evaluator.

Evaluate the learner's written English for the stated real-life situation.

Do not evaluate pronunciation, accent, voice quality or speaking speed.

Score:
1. Grammar - 25
2. Clarity - 25
3. Natural English - 25
4. Situation & Tone - 25

Total = 100.

Use exactly this structure:

## SmartSpeak Score
Total Score: <0-100>/100

- Grammar: <0-25>/25
- Clarity: <0-25>/25
- Natural English: <0-25>/25
- Situation & Tone: <0-25>/25

## What You Did Well
Give short positive points based only on the answer.

## Important Fix
Correct only the most useful mistakes.

## Better Version
Give one improved English version that preserves the learner's meaning.

## One Speaking Tip
Give one short practical tip for the next attempt.

Keep the feedback supportive, specific and concise.
"""

                practice_user = f"""
Situation:
{st.session_state.get(
    "smartspeak_last_situation",
    effective_situation
)}

Requested Tone:
{st.session_state.get(
    "smartspeak_last_tone",
    tone
)}

Original Thought:
{st.session_state.get(
    "smartspeak_original_thought",
    thought
)}

Learner's English Practice:
{practice_text}
"""

                with st.spinner(
                    "SmartSpeak is checking your English..."
                ):
                    try:
                        practice_report = ask_ai(
                            practice_system,
                            practice_user
                        )

                        import re

                        score_patterns = [
                            r"Total Score:\s*(\d{1,3})\s*/\s*100",
                            r"Total Score:\s*(\d{1,3})%",
                            r"SmartSpeak Score:\s*(\d{1,3})"
                        ]

                        # ====================================
                        # SMARTSPEAK SCORE SCALE GUARDRAIL V1
                        # ====================================
                        #
                        # Preserve the score scale whenever the
                        # AI explicitly returns /100 or %.
                        #
                        # Example:
                        #     8/100 -> valid 8
                        #     8%    -> valid 8
                        #
                        # A plain "8" without a scale remains
                        # ambiguous and is rejected.
                        # ====================================

                        smart_score = None
                        _raw_smartspeak_score = None

                        for pattern in score_patterns:
                            match = re.search(
                                pattern,
                                practice_report,
                                re.IGNORECASE
                            )

                            if match:
                                _score_number = match.group(1)

                                _matched_text = match.group(0)

                                if "/100" in _matched_text.replace(" ", ""):
                                    _raw_smartspeak_score = (
                                        f"{_score_number}/100"
                                    )

                                elif "%" in _matched_text:
                                    _raw_smartspeak_score = (
                                        f"{_score_number}%"
                                    )

                                else:
                                    _raw_smartspeak_score = (
                                        _score_number
                                    )

                                break

                        if _raw_smartspeak_score is None:
                            raise ValueError(
                                "SmartSpeak AI response did not "
                                "return a recognizable score."
                            )

                        _smartspeak_score_guard = (
                            validate_score_contract(
                                _raw_smartspeak_score
                            )
                        )

                        if not _smartspeak_score_guard.get(
                            "valid"
                        ):
                            _guard_status = (
                                _smartspeak_score_guard.get(
                                    "status",
                                    "UNKNOWN"
                                )
                            )

                            raise ValueError(
                                "SmartSpeak returned an ambiguous "
                                "or invalid score. "
                                f"Guardrail status: {_guard_status}"
                            )

                        smart_score = int(
                            round(
                                float(
                                    _smartspeak_score_guard[
                                        "score"
                                    ]
                                )
                            )
                        )

                        st.session_state[
                            "smartspeak_practice_report"
                        ] = practice_report

                        st.session_state[
                            "smartspeak_practice_score"
                        ] = smart_score

                        try:
                            save_activity(
                                activity_type="SmartSpeak",
                                activity=(
                                    f"{effective_situation} - "
                                    f"{tone}"
                                ),
                                score=smart_score,
                                review_data={
                                    "title": (
                                        f"SmartSpeak - "
                                        f"{effective_situation}"
                                    ),
                                    "question": (
                                        "Express this thought naturally "
                                        "in English."
                                    ),
                                    "answer": practice_text,
                                    "feedback": practice_report,
                                    "score": smart_score
                                }
                            )

                        except TypeError:
                            save_activity(
                                activity_type="SmartSpeak",
                                activity=(
                                    f"{effective_situation} - "
                                    f"{tone}"
                                ),
                                score=smart_score
                            )

                    except Exception as e:
                        st.error(
                            f"SmartSpeak analysis could not be completed: {e}"
                        )

        practice_report = st.session_state.get(
            "smartspeak_practice_report",
            ""
        )

        practice_score = st.session_state.get(
            "smartspeak_practice_score"
        )

        if practice_report:

            st.markdown("### 📊 SmartSpeak Feedback")

            if practice_score is not None:
                st.metric(
                    "Communication Practice Score",
                    f"{practice_score}%"
                )

            st.markdown(practice_report)

    # --------------------------------------------------------
    # RECENT PRACTICE
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 📌 Recent SmartSpeak Practice")

    db = SessionLocal()

    try:
        smartspeak_logs = (
            db.query(ActivityLog)
            .filter(
                ActivityLog.user_id
                == st.session_state.user_id,
                ActivityLog.activity_type
                == "SmartSpeak"
            )
            .order_by(
                ActivityLog.created_at.desc()
            )
            .limit(10)
            .all()
        )
    finally:
        db.close()

    if smartspeak_logs:

        smart_scores = [
            log.score
            for log in smartspeak_logs
            if log.score is not None
        ]

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Practised",
            len(smartspeak_logs)
        )

        if smart_scores:
            m2.metric(
                "Average",
                f"{round(sum(smart_scores) / len(smart_scores))}%"
            )
            m3.metric(
                "Best",
                f"{max(smart_scores)}%"
            )
        else:
            m2.metric("Average", "—")
            m3.metric("Best", "—")

        for log in smartspeak_logs[:5]:

            score_text = (
                f"{int(log.score)}%"
                if log.score is not None
                else "—"
            )

            st.write(
                f"• **{log.activity}** — {score_text}"
            )

        st.caption(
            "Saved SmartSpeak practice is also available in 📜 History."
        )

    else:
        st.info(
            "Complete your first SmartSpeak practice to start tracking improvement."
        )



# ============================================================
# STEP 6 - COMPLETE PROGRESS INTELLIGENCE
# ============================================================

elif page == "📈 Progress":

    friendly_coach_message("📈 Progress")

    st.title("📈 My Progress")
    st.subheader(f"Progress for {st.session_state.user_name}")
    st.caption("Your progress is calculated from your saved learning activities.")

    db = SessionLocal()
    try:
        progress_logs = (
            db.query(ActivityLog)
            .filter(ActivityLog.user_id == st.session_state.user_id)
            .order_by(ActivityLog.created_at.asc())
            .all()
        )
    finally:
        db.close()

    scored_logs = [log for log in progress_logs if log.score is not None]

    speaking_scores = [
        log.score for log in scored_logs
        if log.activity_type == "Speaking"
    ]
    interview_scores = [
        log.score for log in scored_logs
        if log.activity_type == "Interview"
    ]
    test_scores = [
        log.score for log in scored_logs
        if log.activity_type == "Test"
    ]
    techprep_scores = [
        log.score for log in scored_logs
        if log.activity_type == "TechPrep"
    ]
    smartspeak_progress_scores = [
        log.score for log in scored_logs
        if log.activity_type == "SmartSpeak"
    ]
    assignment_logs = [
        log for log in progress_logs
        if log.activity_type == "Assignment"
    ]

    def safe_average(values):
        return round(sum(values) / len(values)) if values else 0

    overall_scores = (
        speaking_scores
        + interview_scores
        + test_scores
        + techprep_scores
        + smartspeak_progress_scores
    )
    overall_average = safe_average(overall_scores)

    c1, c2, c3 = st.columns(3)
    c1.metric("🗣️ Speaking Avg", f"{safe_average(speaking_scores)}%")
    c2.metric("🎤 Interview Avg", f"{safe_average(interview_scores)}%")
    c3.metric("🧪 Test Avg", f"{safe_average(test_scores)}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("🧑‍💻 TechPrep Avg", f"{safe_average(techprep_scores)}%")
    c5.metric(
        "💬 SmartSpeak Avg",
        f"{safe_average(smartspeak_progress_scores)}%"
    )
    c6.metric("⭐ Overall Avg", f"{overall_average}%")

    st.divider()

    st.subheader("📚 Activity Summary")

    a1, a2, a3 = st.columns(3)
    a1.metric("Speaking", len(speaking_scores))
    a2.metric("Interviews", len(interview_scores))
    a3.metric("Tests", len(test_scores))

    a4, a5, a6 = st.columns(3)
    a4.metric("TechPrep", len(techprep_scores))
    a5.metric("SmartSpeak", len(smartspeak_progress_scores))
    a6.metric("Assignments", len(assignment_logs))

    st.divider()
    st.subheader("📊 Recent Score Trend")

    recent_scored = scored_logs[-10:]

    if recent_scored:
        for log in reversed(recent_scored):
            score = int(log.score)
            label = f"{log.activity_type}: {log.activity}"
            st.write(f"**{label}** — {score}%")
            st.progress(max(0, min(100, score)) / 100)
    else:
        st.info("Complete scored activities to build your progress trend.")

    st.divider()
    st.subheader("🧠 Progress Insight")

    category_data = {
        "Speaking": speaking_scores,
        "Interview": interview_scores,
        "Tests": test_scores,
        "TechPrep": techprep_scores,
        "SmartSpeak": smartspeak_progress_scores,
    }
    available_categories = {
        name: values for name, values in category_data.items() if values
    }

    if available_categories:
        category_averages = {
            name: safe_average(values)
            for name, values in available_categories.items()
        }

        strongest = max(category_averages, key=category_averages.get)
        focus = min(category_averages, key=category_averages.get)

        p1, p2 = st.columns(2)
        p1.success(
            f"💪 Strongest Area: {strongest} "
            f"({category_averages[strongest]}%)"
        )
        p2.info(
            f"🎯 Current Focus: {focus} "
            f"({category_averages[focus]}%)"
        )

        if category_averages[focus] < 50:
            st.write(
                f"Practice **{focus}** with short, simple answers first. "
                "Focus on structure and clarity before adding more detail."
            )
        elif category_averages[focus] < 70:
            st.write(
                f"Your **{focus}** is developing well. "
                "Use examples and review your AI coach feedback before retrying."
            )
        else:
            st.write(
                f"Your recorded areas are progressing well. Keep **{focus}** "
                "consistent while continuing regular practice."
            )
    else:
        st.info(
            "No scored activity is available yet. Complete Speaking, "
            "Interview, Test, TechPrep or SmartSpeak practice to receive "
            "progress insights."
        )

    st.divider()
    st.subheader("📈 Improvement Check")

    if len(overall_scores) >= 2:
        split = max(1, len(overall_scores) // 2)
        earlier = overall_scores[:split]
        recent = overall_scores[split:]

        earlier_avg = safe_average(earlier)
        recent_avg = safe_average(recent) if recent else earlier_avg
        change = recent_avg - earlier_avg

        i1, i2, i3 = st.columns(3)
        i1.metric("Earlier Avg", f"{earlier_avg}%")
        i2.metric("Recent Avg", f"{recent_avg}%")
        i3.metric("Change", f"{change:+d}%")

        if change > 0:
            st.success(f"✅ Your recent average improved by {change}%.")
        elif change == 0:
            st.info(
                "Your average is stable. Continue practice and focus on "
                "the current weak area."
            )
        else:
            st.warning(
                "Your recent average is lower. Review the latest coach feedback "
                "and retry with shorter structured answers."
            )
    else:
        st.info("Complete at least two scored activities to compare improvement.")

    st.divider()
    st.subheader("🎯 Recommended Next Practice")

    if available_categories:
        if focus == "Speaking":
            st.write(
                "Next: Open **🗣️ Speaking**, answer one guided question, "
                "use the Adaptive Coach tip, and retry once."
            )
        elif focus == "Interview":
            st.write(
                "Next: Open **🎤 Interviews** and practice one interview session. "
                "Use direct answer → reason → example → result."
            )
        elif focus == "Tests":
            st.write(
                "Next: Open **🧪 Tests**, review the topic, and attempt another test."
            )
        elif focus == "TechPrep":
            st.write(
                "Next: Open **🧑‍💻 TechPrep**, review one technical question, "
                "structure the concept, and practise your interview answer."
            )
        elif focus == "SmartSpeak":
            st.write(
                "Next: Open **💬 SmartSpeak**, express one real-life thought "
                "in English and practise the improved version."
            )
    else:
        st.write(
            "Start with **🗣️ Speaking Practice** to create your first scored activity."
        )


# ============================================================
# STEP 7 - COMPLETE LEARNING HISTORY
# ============================================================

elif page == "📜 History":

    friendly_coach_message("📜 History")

    st.title("📜 Learning History")
    st.subheader(f"User: {st.session_state.user_id}")
    st.caption("Review your saved activities, scores and full previous work details.")

    db = SessionLocal()
    try:
        history_logs = (
            db.query(ActivityLog)
            .filter(ActivityLog.user_id == st.session_state.user_id)
            .order_by(ActivityLog.created_at.desc())
            .all()
        )
    finally:
        db.close()

    if not history_logs:
        st.info("No learning history is available yet.")
    else:
        all_types = sorted({
            log.activity_type for log in history_logs
            if log.activity_type
        })

        filter_options = ["All"] + all_types
        selected_type = st.selectbox(
            "Filter activity",
            filter_options,
            key="step7_history_filter"
        )

        filtered_logs = (
            history_logs
            if selected_type == "All"
            else [
                log for log in history_logs
                if log.activity_type == selected_type
            ]
        )

        scored_filtered = [
            log for log in filtered_logs
            if log.score is not None
        ]

        st.divider()
        h1, h2, h3, h4 = st.columns(4)

        h1.metric("Total Records", len(filtered_logs))
        h2.metric("Scored Records", len(scored_filtered))

        if scored_filtered:
            avg_history_score = round(
                sum(log.score for log in scored_filtered)
                / len(scored_filtered)
            )
            best_history_score = max(
                log.score for log in scored_filtered
            )
        else:
            avg_history_score = 0
            best_history_score = 0

        h3.metric(
            "Average Score",
            f"{avg_history_score}%"
            if scored_filtered else "—"
        )
        h4.metric(
            "Best Score",
            f"{best_history_score}%"
            if scored_filtered else "—"
        )

        st.divider()
        st.subheader("🕘 Recent Learning Activity")
        st.caption(
            "Open 👁 Review Previous Work to see saved questions, your answers, "
            "correct answers/transcripts, AI feedback and scores for new activities."
        )

        review_map = load_activity_review_map(
            st.session_state.user_id,
            [log.id for log in filtered_logs]
        )

        display_limit = st.selectbox(
            "Show records",
            [10, 20, 50, 100],
            index=1,
            key="step7_history_limit"
        )

        for log in filtered_logs[:display_limit]:
            created_at = log.created_at

            if created_at is not None:
                # Existing DB timestamps are stored as UTC-naive.
                display_time = created_at + timedelta(
                    hours=5,
                    minutes=30
                )
                time_text = display_time.strftime(
                    "%d %b %Y, %I:%M %p"
                )
            else:
                time_text = "Time unavailable"

            score_text = (
                f"{int(log.score)}%"
                if log.score is not None
                else "Completed"
            )

            with st.container(border=True):
                c1, c2 = st.columns([3, 1])

                with c1:
                    st.markdown(
                        f"**{log.activity_type}** — {log.activity}"
                    )
                    st.caption(time_text)

                with c2:
                    st.metric("Result", score_text)

                review = review_map.get(log.id)
                if review:
                    with st.expander("👁 Review Previous Work"):
                        if review.get("title"):
                            st.markdown(f"### {review['title']}")

                        if review.get("topic"):
                            st.write(f"**Topic:** {review['topic']}")
                        if review.get("interview_type"):
                            st.write(f"**Interview Type:** {review['interview_type']}")
                        if review.get("difficulty"):
                            st.write(f"**Difficulty:** {review['difficulty']}")
                        if review.get("mode"):
                            st.write(f"**Mode:** {review['mode']}")

                        if review.get("question"):
                            st.markdown("**Question / Task**")
                            st.write(review["question"])

                        saved_answer = review.get(
                            "your_answer",
                            review.get("answer")
                        )

                        if saved_answer is not None:
                            st.markdown("**Your Answer**")
                            st.write(saved_answer)

                        if review.get("transcript"):
                            st.markdown("**Your Transcript**")
                            st.write(review["transcript"])

                        if review.get("correct_answer") is not None:
                            st.markdown("**Correct / Better Answer**")
                            st.write(review["correct_answer"])

                        if review.get("result"):
                            st.write(f"**Result:** {review['result']}")

                        if review.get("questions"):
                            st.markdown("**Question-by-Question Review**")
                            for qn, item in enumerate(review["questions"], 1):
                                with st.container(border=True):
                                    st.markdown(f"**Q{qn}. {item.get('question', '')}**")
                                    st.write(f"Your answer: {item.get('selected', '')}")
                                    st.write(f"Correct answer: {item.get('correct', '')}")
                                    st.write(
                                        "Result: " +
                                        ("✅ Correct" if item.get("is_correct") else "❌ Needs Review")
                                    )
                                    if item.get("explanation"):
                                        st.caption(item["explanation"])

                        saved_feedback = review.get(
                            "ai_feedback",
                            review.get("feedback")
                        )

                        if saved_feedback:
                            st.markdown("**AI Feedback**")
                            st.markdown(saved_feedback)

                        if review.get("score") is not None:
                            st.metric("Saved Score", f"{int(review['score'])}%")
                else:
                    with st.expander("👁 Review Previous Work"):
                        st.info(
                            "This is an older activity. Its score/history is available, "
                            "but the full question, answer or AI feedback was not saved "
                            "when it was originally completed. New activities completed "
                            "after this update will include full review details."
                        )

        st.divider()
        st.subheader("📊 History Insight")

        scored_all = [
            log for log in history_logs
            if log.score is not None
        ]

        if scored_all:
            type_scores = {}

            for log in scored_all:
                type_scores.setdefault(
                    log.activity_type,
                    []
                ).append(log.score)

            type_averages = {
                activity_type: round(
                    sum(scores) / len(scores)
                )
                for activity_type, scores in type_scores.items()
            }

            strongest_history = max(
                type_averages,
                key=type_averages.get
            )
            focus_history = min(
                type_averages,
                key=type_averages.get
            )

            i1, i2 = st.columns(2)
            i1.success(
                f"💪 Best Recorded Area: "
                f"{strongest_history} "
                f"({type_averages[strongest_history]}%)"
            )
            i2.info(
                f"🎯 Practice Focus: "
                f"{focus_history} "
                f"({type_averages[focus_history]}%)"
            )

            recent_five = scored_all[:5]
            older_five = scored_all[5:10]

            if recent_five and older_five:
                recent_avg = round(
                    sum(log.score for log in recent_five)
                    / len(recent_five)
                )
                older_avg = round(
                    sum(log.score for log in older_five)
                    / len(older_five)
                )
                history_change = recent_avg - older_avg

                st.write(
                    f"Recent 5 scored activities: **{recent_avg}%** | "
                    f"Previous 5: **{older_avg}%** | "
                    f"Change: **{history_change:+d}%**"
                )

                if history_change > 0:
                    st.success(
                        "✅ Your recent recorded performance is improving."
                    )
                elif history_change == 0:
                    st.info(
                        "Your recent recorded performance is stable."
                    )
                else:
                    st.warning(
                        "Your recent recorded average is lower. "
                        "Review your latest coach feedback before retrying."
                    )
        else:
            st.info(
                "Complete scored activities to receive history insights."
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI English Coach | "
    "Learn → Practice → Analyze → Improve → Retry"
)
