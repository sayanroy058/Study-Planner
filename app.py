import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, flash, jsonify, redirect, render_template,
    request, session, url_for,
)
from flask_login import (
    LoginManager, UserMixin, current_user,
    login_required, login_user, logout_user,
)
from PyPDF2 import PdfReader
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import docx
import google.generativeai as genai


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "studyplanner.db"

ALLOWED_EXTENSIONS = {"txt", "md", "pdf", "docx"}
MAX_TEXT_CHARS = 20000

MODEL_NAME = "gemini-3.1-flash-lite-preview"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT      NOT NULL UNIQUE,
            email         TEXT      NOT NULL UNIQUE,
            password_hash TEXT      NOT NULL,
            created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_items (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            item_type  TEXT    NOT NULL,
            title      TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            action     TEXT    NOT NULL,
            detail     TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def _is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _extract_text_from_response(response) -> str:
    if getattr(response, "text", None):
        return response.text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        texts = [getattr(part, "text", "") for part in parts if getattr(part, "text", None)]
        joined = "\n".join(texts).strip()
        if joined:
            return joined

    return ""


def _generate_text(prompt: str, temperature: float = 0.4) -> tuple[str, str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Add it in your environment or .env file.")

    genai.configure(api_key=api_key)

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config={"temperature": temperature},
        )
        response = model.generate_content(prompt)
        text = _extract_text_from_response(response).strip()
        if not text:
            raise RuntimeError("Model returned an empty response.")
        return text, MODEL_NAME
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Gemini request failed for {MODEL_NAME}: {exc}") from exc


def _extract_text_from_file(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if extension == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if extension == ".docx":
        document = docx.Document(str(file_path))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        return "\n".join(paragraphs)

    raise ValueError("Unsupported file format")


def _planner_prompt(payload: dict) -> str:
    focus_areas = payload.get("focusAreas", [])
    if isinstance(focus_areas, str):
        focus_areas = [area.strip() for area in focus_areas.split(",") if area.strip()]

    return f"""
You are an elite learning coach and productivity mentor.
Create a personalized, practical study plan.

Student details:
- Goal: {payload.get('goal', 'Not provided')}
- Current level: {payload.get('currentLevel', 'Not provided')}
- Hours per day available: {payload.get('hoursPerDay', 'Not provided')}
- Days per week available: {payload.get('daysPerWeek', 'Not provided')}
- Target date: {payload.get('targetDate', 'Not provided')}
- Focus areas: {', '.join(focus_areas) if focus_areas else 'None'}
- Notes: {payload.get('notes', 'None')}

Output format (markdown):
1) Clear strategy summary (5-7 lines)
2) Weekly roadmap table
3) Daily time-block template
4) Milestone checkpoints and self-tests
5) Recommended resources and practice style
6) Two fallback options for low-energy days

Keep it realistic and adaptive.
""".strip()


def _chat_prompt(payload: dict) -> str:
    return f"""
You are Study Planner AI Assistant.
Answer the user doubt clearly with a teacher-like style.

User question:
{payload.get('question', '')}

Extra context from user:
{payload.get('context', 'None')}

Response requirements:
- Explain in simple language
- Give one small example
- End with one quick practice task
""".strip()


def _quiz_prompt(payload: dict) -> str:
    try:
        question_count = int(payload.get("questionCount", 8))
    except (TypeError, ValueError):
        question_count = 8
    question_count = max(3, min(question_count, 20))

    return f"""
You are a quiz generator for students.
Create a mixed quiz in markdown.

Details:
- Topic: {payload.get('topic', 'General')}
- Difficulty: {payload.get('difficulty', 'medium')}
- Questions needed: {question_count}

Rules:
- 60% conceptual MCQ, 40% short answer.
- For MCQ include 4 options each.
- After questions, include an answer key and short explanation.
- End with a score interpretation guide out of {question_count}.
""".strip()


def _flashcards_prompt(payload: dict) -> str:
    try:
        card_count = int(payload.get("cardCount", 10))
    except (TypeError, ValueError):
        card_count = 10
    card_count = max(4, min(card_count, 25))

    return f"""
You are an expert flashcard creator for learners.

Create exactly {card_count} flashcards in markdown table format.

Details:
- Topic: {payload.get('topic', 'General')}
- Difficulty: {payload.get('difficulty', 'medium')}

Output format:
| # | Question | Answer |
|---|----------|--------|

Rules:
- Keep question text crisp.
- Keep answer concise but complete.
- Cover fundamentals and commonly tested points.
""".strip()


def _revision_plan_prompt(payload: dict) -> str:
    weak_areas = payload.get("weakAreas", "None")
    return f"""
You are a revision strategist.

Create a 7-day adaptive revision plan.

Input:
- Primary topic: {payload.get('topic', 'General')}
- Weak areas: {weak_areas}
- Time available per day: {payload.get('hoursPerDay', '2')} hours

Output in markdown:
1) Diagnostic overview
2) 7-day revision table (day, focus, method, expected outcome)
3) Active recall blocks and spaced repetition checkpoints
4) One mock test strategy for day 7
""".strip()


def _summary_prompt(extracted_text: str, filename: str) -> str:
    text = extracted_text.strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n\n[Truncated for processing]"

    return f"""
You are an academic summarization assistant.
Summarize the uploaded document in student-friendly format.

Document name: {filename}

Document text:
{text}

Output as markdown with these sections:
1) Quick Summary
2) Key Concepts
3) Important Terms and Definitions
4) Exam-Oriented Questions (5)
5) Revision Checklist
""".strip()


# ---------------------------------------------------------------------------
# Flask app + Flask-Login setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.secret_key = os.getenv("SECRET_KEY", "studyforge_dev_secret_change_in_prod")

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"


class User(UserMixin):
    def __init__(self, row):
        self.id = row["id"]
        self.username = row["username"]
        self.email = row["email"]


@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return User(row) if row else None


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        conn = get_db()
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier),
        ).fetchone()
        conn.close()

        if row and check_password_hash(row["password_hash"], password):
            login_user(User(row), remember=remember)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard"))

        flash("Invalid username/email or password.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        error = None
        if not username or len(username) < 3:
            error = "Username must be at least 3 characters."
        elif not email or "@" not in email:
            error = "Please enter a valid email address."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."

        if not error:
            conn = get_db()
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ? OR email = ?",
                (username, email),
            ).fetchone()
            if existing:
                error = "Username or email already registered."
                conn.close()
            else:
                hashed = generate_password_hash(password)
                conn.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, hashed),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM users WHERE username = ?", (username,)
                ).fetchone()
                conn.close()
                login_user(User(row))
                flash("Account created! Welcome to StudyForge AI.", "success")
                return redirect(url_for("dashboard"))

        if error:
            flash(error, "error")

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# Page routes (protected)
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/planner")
@login_required
def planner():
    return render_template("planner.html")


@app.route("/quiz")
@login_required
def quiz():
    return render_template("quiz.html")


@app.route("/flashcards")
@login_required
def flashcards():
    return render_template("flashcards.html")


@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html")


@app.route("/summarizer")
@login_required
def summarizer():
    return render_template("summarizer.html")


@app.route("/revision")
@login_required
def revision():
    return render_template("revision.html")


# ---------------------------------------------------------------------------
# API routes (protected)
# ---------------------------------------------------------------------------

@app.route("/api/study-plan", methods=["POST"])
@login_required
def create_study_plan():
    try:
        payload = request.get_json(silent=True) or {}
        goal = str(payload.get("goal", "")).strip()
        if not goal:
            return jsonify({"error": "Goal is required."}), 400

        prompt = _planner_prompt(payload)
        answer, model_used = _generate_text(prompt, temperature=0.35)
        return jsonify({"result": answer, "model": model_used})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chat", methods=["POST"])
@login_required
def chat_with_ai():
    try:
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Question is required."}), 400

        prompt = _chat_prompt(payload)
        answer, model_used = _generate_text(prompt, temperature=0.45)
        return jsonify({"result": answer, "model": model_used})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/quiz", methods=["POST"])
@login_required
def generate_quiz():
    try:
        payload = request.get_json(silent=True) or {}
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            return jsonify({"error": "Topic is required."}), 400

        prompt = _quiz_prompt(payload)
        answer, model_used = _generate_text(prompt, temperature=0.5)
        return jsonify({"result": answer, "model": model_used})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/flashcards", methods=["POST"])
@login_required
def generate_flashcards():
    try:
        payload = request.get_json(silent=True) or {}
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            return jsonify({"error": "Topic is required."}), 400

        prompt = _flashcards_prompt(payload)
        answer, model_used = _generate_text(prompt, temperature=0.45)
        return jsonify({"result": answer, "model": model_used})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/revision-plan", methods=["POST"])
@login_required
def generate_revision_plan():
    try:
        payload = request.get_json(silent=True) or {}
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            return jsonify({"error": "Topic is required."}), 400

        prompt = _revision_plan_prompt(payload)
        answer, model_used = _generate_text(prompt, temperature=0.35)
        return jsonify({"result": answer, "model": model_used})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/upload-summary", methods=["POST"])
@login_required
def upload_and_summarize():
    temp_path = None
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not _is_allowed_file(file.filename):
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            return jsonify({"error": f"Unsupported file type. Allowed: {allowed}"}), 400

        safe_name = secure_filename(file.filename)
        unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe_name}"
        temp_path = UPLOAD_DIR / unique_name
        file.save(temp_path)

        extracted_text = _extract_text_from_file(temp_path)
        if not extracted_text.strip():
            return jsonify({"error": "Could not read meaningful text from the uploaded file."}), 400

        prompt = _summary_prompt(extracted_text, safe_name)
        answer, model_used = _generate_text(prompt, temperature=0.25)
        return jsonify({"result": answer, "model": model_used})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

with app.app_context():
    init_db()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
