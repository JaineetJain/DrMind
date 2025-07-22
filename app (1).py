import os
import json
import random
from datetime import datetime
from flask import Flask, request, render_template_string, flash, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from textblob import TextBlob
import google.generativeai as genai
import requests

app = Flask(__name__)
app.config["SECRET_KEY"] = "dr_mind_secret_key_2024"

# Database configuration for Render PostgreSQL
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    # Fix for Render's PostgreSQL URL format
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
else:
    # Fallback to SQLite for local development
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///drmind.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --- AI API KEYS (SECURELY FROM ENVIRONMENT VARIABLES) ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# Configure Google AI
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Google AI configuration error: {e}")

class MoodEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mood = db.Column(db.String(50), nullable=False)
    journal = db.Column(db.String(1000))
    sentiment = db.Column(db.Float)
    comfort_message = db.Column(db.String(500))
    suggestions = db.Column(db.String(1000))
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"User('{self.email}')"

# Comprehensive mood options
MOODS = [
    {"emoji": "😃", "label": "Joyful", "value": 0.9},
    {"emoji": "😊", "label": "Content", "value": 0.7},
    {"emoji": "😌", "label": "Peaceful", "value": 0.6},
    {"emoji": "😇", "label": "Grateful", "value": 0.8},
    {"emoji": "🤗", "label": "Loved", "value": 0.8},
    {"emoji": "🥳", "label": "Excited", "value": 0.9},
    {"emoji": "😎", "label": "Confident", "value": 0.7},
    {"emoji": "😋", "label": "Satisfied", "value": 0.6},
    {"emoji": "😤", "label": "Determined", "value": 0.5},
    {"emoji": "😐", "label": "Neutral", "value": 0.0},
    {"emoji": "😕", "label": "Confused", "value": -0.2},
    {"emoji": "😟", "label": "Worried", "value": -0.4},
    {"emoji": "😢", "label": "Sad", "value": -0.6},
    {"emoji": "😞", "label": "Down", "value": -0.7},
    {"emoji": "😩", "label": "Exhausted", "value": -0.5},
    {"emoji": "😡", "label": "Angry", "value": -0.8},
    {"emoji": "😱", "label": "Anxious", "value": -0.6},
    {"emoji": "😖", "label": "Stressed", "value": -0.5},
    {"emoji": "😰", "label": "Overwhelmed", "value": -0.7},
    {"emoji": "😭", "label": "Devastated", "value": -0.9}
]

# Motivational quotes
QUOTES = [
    "Every day may not be good, but there is something good in every day.",
    "You are stronger than you think.",
    "Progress, not perfection.",
    "Small steps every day lead to big changes.",
    "You have overcome challenges before, you can do it again.",
    "Your feelings are valid. Take it one day at a time.",
    "Celebrate your wins, no matter how small.",
    "You are not alone. Reach out if you need support.",
    "Rest is productive, too.",
    "Be gentle with yourself.",
    "The only way to do great work is to love what you do.",
    "Believe you can and you're halfway there.",
    "It's okay to not be okay.",
    "Your mental health is a priority.",
    "You are worthy of love and respect."
]

def get_huggingface_ai_response(mood, journal, sentiment):
    """Get AI response using Hugging Face Inference API (FREE)"""
    try:
        API_URL = "https://router.huggingface.co/v1/chat/completions"
        
        prompt = f"You are Dr. Mind, a compassionate AI mental health companion. The user wrote: \"{journal}\" and selected the mood: \"{mood}\" with a sentiment score of {sentiment:.2f}. Please provide a warm, empathetic comfort message (2-3 sentences) and three actionable, practical suggestions for what to do next. Format your response as: COMFORT: [your comfort message here] SUGGESTIONS: - [suggestion 1] - [suggestion 2] - [suggestion 3]"
        messages = [{"role": "user", "content": prompt}]
        
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "messages": messages}
        
        response = requests.post(API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and result.get("choices"):
                ai_response = result["choices"][0]["message"]["content"]
                
                lines = ai_response.split('\n')
                comfort = ""
                suggestions = []
                
                for line in lines:
                    if 'COMFORT:' in line:
                        comfort = line.replace('COMFORT:', '').strip()
                    elif line.strip().startswith('-') or line.strip().startswith('•'):
                        suggestions.append(line.strip().lstrip('- ').lstrip('• '))
                
                if not comfort:
                    comfort = f"I understand you're feeling {mood.lower()}. Your feelings are valid and important."
                
                if not suggestions:
                    suggestions = [
                        "Take a moment to breathe deeply and center yourself",
                        "Write down your thoughts to help process them",
                        "Reach out to someone you trust for support"
                    ]
                
                return {'comfort': comfort, 'suggestions': suggestions[:3]}
        
        print("🤖 Hugging Face API failed, using fallback system...")
        return get_fallback_response(mood, journal, sentiment)
        
    except Exception as e:
        print(f"🤖 Hugging Face API error: {e}")
        return get_fallback_response(mood, journal, sentiment)

def get_fallback_response(mood, journal, sentiment):
    """Fallback response when AI APIs are not available"""
    
    comfort_messages = {
        'positive': [
            "It's wonderful to see you in such a positive space! Your energy is contagious and inspiring.",
            "Your positive outlook is truly beautiful. This kind of energy can create amazing ripples in your life.",
            "What a joy to witness your happiness! These moments of positivity are precious and worth celebrating."
        ],
        'neutral': [
            "It's perfectly okay to feel neutral. Every emotion has its place in our journey.",
            "Neutral moments are often when we can best observe and understand ourselves.",
            "There's wisdom in accepting all our emotional states, including the calm neutral ones."
        ],
        'negative': [
            "I hear you, and your feelings are completely valid. It's okay to not be okay.",
            "Your emotions are real and important. You don't have to rush through this difficult time.",
            "It takes courage to acknowledge when we're struggling. You're showing strength by being honest."
        ]
    }
    
    suggestions_database = {
        'positive': [
            "Share this positive energy with someone who might need it",
            "Document this feeling to remember it during tougher times",
            "Use this momentum to tackle something you've been putting off"
        ],
        'neutral': [
            "Take a moment to practice mindfulness or meditation",
            "Try a new activity to add some variety to your day",
            "Connect with a friend or family member"
        ],
        'negative': [
            "Practice self-compassion - be as kind to yourself as you would be to a friend",
            "Try some gentle physical activity like walking or stretching",
            "Consider talking to someone you trust about how you're feeling"
        ]
    }
    
    if sentiment > 0.3:
        category = 'positive'
    elif sentiment < -0.3:
        category = 'negative'
    else:
        category = 'neutral'
    
    comfort = random.choice(comfort_messages[category])
    final_suggestions = suggestions_database[category]
    
    return {'comfort': comfort, 'suggestions': final_suggestions}

def get_ai_response(mood, journal, sentiment):
    """Get AI-generated comfort message and suggestions"""
    return get_huggingface_ai_response(mood, journal, sentiment)

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    try:
        analysis = TextBlob(text)
        sentiment = analysis.sentiment.polarity
        return max(-1, min(1, sentiment))
    except Exception:
        return 0.0

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        mood = request.form.get('mood')
        journal = request.form.get('journal')
        
        if not mood or not journal:
            flash('⚠️ Please select a mood and write a journal entry!', 'error')
        else:
            try:
                sentiment = analyze_sentiment(journal)
                ai_response = get_ai_response(mood, journal, sentiment)
                
                new_entry = MoodEntry(
                    mood=mood,
                    journal=journal,
                    sentiment=sentiment,
                    comfort_message=ai_response['comfort'],
                    suggestions=json.dumps(ai_response['suggestions'])
                )
                
                db.session.add(new_entry)
                db.session.commit()
                
                flash('📝 Entry saved successfully!', 'success')
                return redirect(url_for('index'))
                
            except Exception as e:
                flash(f'⚠️ Error saving entry: {str(e)}', 'error')
    
    entries = MoodEntry.query.order_by(MoodEntry.date.desc()).all()
    
    chart_data = []
    for entry in entries[-10:]:
        chart_data.append({
            'date': entry.date.strftime('%Y-%m-%d'),
            'sentiment': entry.sentiment
        })
    
    total_entries = len(entries)
    if total_entries > 0:
        avg_sentiment = sum(entry.sentiment for entry in entries) / total_entries
        positive_days = len([e for e in entries if e.sentiment > 0.3])
    else:
        avg_sentiment = 0
        positive_days = 0
    
    current_quote = random.choice(QUOTES)
    
    return render_template_string(HTML_TEMPLATE, 
                                moods=MOODS, 
                                entries=entries, 
                                chart_data=json.dumps(chart_data),
                                total_entries=total_entries,
                                avg_sentiment=avg_sentiment,
                                positive_days=positive_days,
                                current_quote=current_quote)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not (first_name and last_name and email and password and confirm_password):
            flash("⚠️ All fields are required!", "error")
            return render_template_string(REGISTER_TEMPLATE, style_block=STYLE_BLOCK)

        if password != confirm_password:
            flash("⚠️ Passwords do not match!", "error")
            return render_template_string(REGISTER_TEMPLATE, style_block=STYLE_BLOCK)

        if len(password) < 8:
            flash("⚠️ Password must be at least 8 characters long!", "error")
            return render_template_string(REGISTER_TEMPLATE, style_block=STYLE_BLOCK)

        if not any(char.isupper() for char in password):
            flash("⚠️ Password must contain at least one uppercase letter!", "error")
            return render_template_string(REGISTER_TEMPLATE, style_block=STYLE_BLOCK)

        if not any(char.islower() for char in password):
            flash("⚠️ Password must contain at least one lowercase letter!", "error")
            return render_template_string(REGISTER_TEMPLATE, style_block=STYLE_BLOCK)

        if not any(char.isdigit() for char in password):
            flash("⚠️ Password must contain at least one number!", "error")
            return render_template_string(REGISTER_TEMPLATE, style_block=STYLE_BLOCK)

        if not any(char in '!@#$%^&*()_+-=[]{}|;:",.<>/?' for char in password):
            flash("⚠️ Password must contain at least one special character!", "error")
            return render_template_string(REGISTER_TEMPLATE, style_block=STYLE_BLOCK)

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("⚠️ Email already registered! Please log in.", "error")
            return render_template_string(REGISTER_TEMPLATE, style_block=STYLE_BLOCK)

        new_user = User(first_name=first_name, last_name=last_name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for("login"))
    return render_template_string(REGISTER_TEMPLATE, style_block=STYLE_BLOCK)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session["user"] = user.email
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials. Please try again.", "error")
    return render_template_string(LOGIN_TEMPLATE, style_block=STYLE_BLOCK)

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

@app.route('/test/huggingface')
def test_huggingface():
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "Say hello from Hugging Face!"}]}
        resp = requests.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=data)
        return f"Hugging Face response: {resp.json()}"
    except Exception as e:
        return f"Hugging Face error: {e}", 500

@app.route('/test/cohere')
def test_cohere():
    try:
        headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "command", "prompt": "Say hello from Cohere!", "max_tokens": 20}
        resp = requests.post("https://api.cohere.ai/v1/generate", headers=headers, json=data)
        return f"Cohere response: {resp.json()}"
    except Exception as e:
        return f"Cohere error: {e}", 500

# Custom Jinja filter
@app.template_filter('from_json')
def from_json(value):
    try:
        return json.loads(value)
    except:
        return []

# Error handlers for better debugging
@app.errorhandler(500)
def internal_error(error):
    print(f"500 Internal Server Error: {error}")
    return f"Internal Server Error: {error}", 500

@app.errorhandler(404)
def not_found_error(error):
    print(f"404 Not Found: {error}")
    return f"Page Not Found: {error}", 404

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled Exception: {e}")
    return f"An error occurred: {e}", 500

# HTML Templates
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dr. Mind - AI-Powered Mood Tracker</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; min-height: 100vh; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); overflow-x: hidden; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; min-height: 100vh; }
        .header { position: relative; text-align: center; margin-bottom: 30px; color: white; }
        .profile-menu { position: absolute; top: 20px; right: 20px; z-index: 1000; }
        .profile-button { background: rgba(255, 255, 255, 0.2); color: white; border: none; padding: 10px 15px; border-radius: 20px; cursor: pointer; font-size: 1rem; font-weight: 600; transition: background 0.3s ease; }
        .profile-button:hover { background: rgba(255, 255, 255, 0.3); }
        .profile-dropdown-content { display: none; position: absolute; background-color: rgba(255, 255, 255, 0.95); min-width: 160px; box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2); z-index: 1; border-radius: 10px; right: 0; margin-top: 10px; overflow: hidden; }
        .profile-dropdown-content a { color: #333; padding: 12px 16px; text-decoration: none; display: block; font-size: 0.95rem; transition: background-color 0.3s ease; }
        .profile-dropdown-content a:hover { background-color: #f1f1f1; }
        .profile-menu:hover .profile-dropdown-content { display: block; }
        .header h1 { font-size: 3rem; font-weight: 700; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .main-card { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .mood-section { margin-bottom: 30px; }
        .mood-section h2 { color: #333; margin-bottom: 20px; font-size: 1.8rem; text-align: center; }
        .mood-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .mood-option { display: flex; flex-direction: column; align-items: center; padding: 15px; border: 2px solid transparent; border-radius: 15px; cursor: pointer; transition: all 0.3s ease; background: rgba(255,255,255,0.8); }
        .mood-option:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
        .mood-option.selected { border-color: #667eea; background: linear-gradient(135deg, #667eea, #764ba2); color: white; transform: scale(1.05); }
        .mood-emoji { font-size: 2.5rem; margin-bottom: 8px; }
        .mood-label { font-size: 0.9rem; font-weight: 500; text-align: center; }
        .journal-section { margin-bottom: 30px; }
        .journal-section h2 { color: #333; margin-bottom: 15px; font-size: 1.8rem; text-align: center; }
        .journal-input { width: 100%; max-width: 400px; min-height: 40px; padding: 10px 15px; border: 2px solid #e1e5e9; border-radius: 10px; font-size: 1rem; font-family: inherit; resize: vertical; transition: border-color 0.3s ease; background: rgba(255,255,255,0.9); margin: 0 auto; display: block; }
        .journal-input:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
        .submit-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 30px; border-radius: 25px; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; display: block; margin: 20px auto; min-width: 200px; }
        .submit-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3); }
        .flash-message { padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center; font-weight: 500; }
        .flash-success { background: linear-gradient(135deg, #43e97b, #38f9d7); color: white; }
        .flash-error { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
        .entry-card { background: rgba(255,255,255,0.9); border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .entry-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .entry-mood { font-size: 2rem; }
        .entry-date { color: #666; font-size: 0.9rem; }
        .entry-sentiment { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 5px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: 500; margin-bottom: 10px; display: inline-block; }
        .entry-text { color: #333; line-height: 1.6; margin-bottom: 15px; }
        .entry-suggestions { background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #667eea; }
        .entry-suggestions h4 { color: #667eea; margin-bottom: 10px; font-size: 1rem; }
        .entry-suggestions ul { list-style: none; padding: 0; }
        .entry-suggestions li { margin: 5px 0; padding-left: 20px; position: relative; }
        .entry-suggestions li:before { content: "•"; color: #667eea; position: absolute; left: 0; }
        .chart-container { background: rgba(255,255,255,0.9); border-radius: 15px; padding: 20px; margin: 30px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1); height: 300px; }
        .chart-container h2 { color: #333; margin-bottom: 20px; text-align: center; font-size: 1.8rem; }
        .stats-section { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .stat-card { background: rgba(255,255,255,0.9); border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .stat-number { font-size: 2.5rem; font-weight: 700; color: #667eea; margin-bottom: 5px; }
        .stat-label { color: #666; font-size: 0.9rem; }
        .motivational-quote { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 25px; border-radius: 15px; text-align: center; margin: 30px 0; font-style: italic; font-size: 1.2rem; }
        .entries-section { margin-top: 40px; }
        .entries-section h2 { color: #333; margin-bottom: 20px; font-size: 1.8rem; text-align: center; }
        @media (max-width: 768px) { .container { padding: 15px; } .header h1 { font-size: 2rem; } .mood-grid { grid-template-columns: repeat(auto-fit, minmax(70px, 1fr)); gap: 10px; } .mood-emoji { font-size: 2rem; } .main-card { padding: 20px; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Dr. Mind</h1>
            <p>AI-Powered Mood Tracking & Wellness Companion</p>
            <div class="profile-menu">
                <button class="profile-button">👤 Profile</button>
                <div class="profile-dropdown-content">
                    <a href="#">My Profile</a>
                    <a href="#">Settings</a>
                    <a href="/logout">Logout</a>
                </div>
            </div>
        </div>

        <div class="main-card">
            <div class="mood-section">
                <h2>How are you feeling today?</h2>
                <form method="POST">
                    <div class="mood-grid">
                        {% for mood in moods %}
                        <label class="mood-option" onclick="selectMood(this, '{{ mood.label }}')">
                            <div class="mood-emoji">{{ mood.emoji }}</div>
                            <div class="mood-label">{{ mood.label }}</div>
                            <input type="radio" name="mood" value="{{ mood.label }}" style="display: none;">
                        </label>
                        {% endfor %}
                    </div>
                    
                    <div class="journal-section">
                        <h2>Share your thoughts...</h2>
                        <textarea class="journal-input" name="journal" placeholder="Write about your day, your feelings, or anything on your mind..." required></textarea>
                    </div>

                    <button type="submit" class="submit-btn">📝 Save Entry & Get AI Insights</button>
                </form>
            </div>

            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="flash-message flash-{{ 'success' if category == 'success' else 'error' }}">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
        </div>

        <div class="chart-container">
            <h2>Your Mood Journey</h2>
            <canvas id="moodChart"></canvas>
        </div>

        <div class="stats-section">
            <div class="stat-card">
                <div class="stat-number">{{ total_entries }}</div>
                <div class="stat-label">Total Entries</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ "%.2f"|format(avg_sentiment) }}</div>
                <div class="stat-label">Average Sentiment</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ positive_days }}</div>
                <div class="stat-label">Positive Days</div>
            </div>
        </div>

        <div class="motivational-quote">"{{ current_quote }}"</div>

        <div class="entries-section">
            <h2>Your Journal Entries</h2>
            {% if entries %}
                {% for entry in entries %}
                <div class="entry-card">
                    <div class="entry-header">
                        <div class="entry-mood">
                            {% for mood in moods %}
                                {% if mood.label == entry.mood %}{{ mood.emoji }}{% endif %}
                            {% endfor %}
                        </div>
                        <div class="entry-date">{{ entry.date.strftime('%Y-%m-%d %H:%M') }}</div>
                    </div>
                    <div class="entry-sentiment">Sentiment: {{ "%.2f"|format(entry.sentiment) }}</div>
                    <div class="entry-text">{{ entry.journal }}</div>
                    <div class="entry-suggestions">
                        <h4>🤖 AI Response:</h4>
                        <p><strong>Comfort:</strong> {{ entry.comfort_message }}</p>
                        <h4>Suggestions:</h4>
                        <ul>
                            {% for suggestion in entry.suggestions|from_json %}
                                <li>{{ suggestion }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <p style="text-align: center; color: #666; font-style: italic;">No entries yet. Start your journey by adding your first entry!</p>
            {% endif %}
        </div>
    </div>

    <script>
        function selectMood(element, mood) {
            document.querySelectorAll('.mood-option').forEach(option => { option.classList.remove('selected'); });
            element.classList.add('selected');
            const radio = element.querySelector('input[type="radio"]');
            radio.checked = true;
        }

        const chartData = {{ chart_data|safe }};
        if (chartData.length > 0) {
            const ctx = document.getElementById('moodChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.map(item => item.date),
                    datasets: [{
                        label: 'Mood Sentiment',
                        data: chartData.map(item => item.sentiment),
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#667eea',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2,
                        pointRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: false, min: -1, max: 1, ticks: { stepSize: 0.25 } },
                        x: { ticks: { maxRotation: 45, minRotation: 45 } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }
    </script>
</body>
</html>'''

LOGIN_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Dr. Mind</title>
    <style>{{style_block}}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Dr. Mind</h1>
            <p>Login to your account</p>
        </div>
        <div class="main-card">
            <form method="POST">
                <div class="journal-section">
                    <h2>Email</h2>
                    <input type="email" name="email" class="journal-input" placeholder="Enter your email" required />
                </div>
                <div class="journal-section">
                    <h2>Password</h2>
                    <input type="password" name="password" class="journal-input" placeholder="Enter your password" required />
                </div>
                <button type="submit" class="submit-btn">🔐 Login</button>
            </form>
            <p style="text-align:center;margin-top:20px;">Don't have an account? <a href="/register">Register</a></p>
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="flash-message flash-{{ 'success' if category == 'success' else 'error' }}">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
        </div>
    </div>
</body>
</html>'''

REGISTER_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register - Dr. Mind</title>
    <style>{{style_block}}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Dr. Mind</h1>
            <p>Create a new account</p>
        </div>
        <div class="main-card">
            <form method="POST">
                <div class="journal-section">
                    <h2>First Name</h2>
                    <input type="text" name="first_name" class="journal-input" placeholder="Enter your first name" required />
                </div>
                <div class="journal-section">
                    <h2>Last Name</h2>
                    <input type="text" name="last_name" class="journal-input" placeholder="Enter your last name" required />
                </div>
                <div class="journal-section">
                    <h2>Email</h2>
                    <input type="email" name="email" class="journal-input" placeholder="Enter your email" required />
                </div>
                <div class="journal-section">
                    <h2>Password</h2>
                    <input type="password" name="password" id="password" class="journal-input" placeholder="Create a password" required />
                    <div class="password-requirements" style="text-align: left; margin-top: 10px; font-size: 0.9em; color: #555;">
                        <p id="req-length"><span class="tick-icon">❌</span> Minimum 8 characters</p>
                        <p id="req-uppercase"><span class="tick-icon">❌</span> At least one uppercase letter (A–Z)</p>
                        <p id="req-lowercase"><span class="tick-icon">❌</span> At least one lowercase letter (a–z)</p>
                        <p id="req-number"><span class="tick-icon">❌</span> At least one number (0–9)</p>
                        <p id="req-special"><span class="tick-icon">❌</span> At least one special character (!@#$%^&* etc.)</p>
                    </div>
                </div>
                <div class="journal-section">
                    <h2>Confirm Password</h2>
                    <input type="password" name="confirm_password" class="journal-input" placeholder="Confirm your password" required />
                </div>
                <button type="submit" class="submit-btn">📝 Register</button>
            </form>
            <p style="text-align:center;margin-top:20px;">Already have an account? <a href="/login">Login</a></p>
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="flash-message flash-{{ 'success' if category == 'success' else 'error' }}">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
        </div>
    </div>
    <script>
        const passwordInput = document.getElementById("password");
        const reqLength = document.getElementById("req-length");
        const reqUppercase = document.getElementById("req-uppercase");
        const reqLowercase = document.getElementById("req-lowercase");
        const reqNumber = document.getElementById("req-number");
        const reqSpecial = document.getElementById("req-special");

        passwordInput.addEventListener("keyup", function() {
            const password = passwordInput.value;
            if (password.length >= 8) {
                reqLength.innerHTML = '<span class="tick-icon">✅</span> Minimum 8 characters';
            } else {
                reqLength.innerHTML = '<span class="tick-icon">❌</span> Minimum 8 characters';
            }
            if (/[A-Z]/.test(password)) {
                reqUppercase.innerHTML = '<span class="tick-icon">✅</span> At least one uppercase letter (A–Z)';
            } else {
                reqUppercase.innerHTML = '<span class="tick-icon">❌</span> At least one uppercase letter (A–Z)';
            }
            if (/[a-z]/.test(password)) {
                reqLowercase.innerHTML = '<span class="tick-icon">✅</span> At least one lowercase letter (a–z)';
            } else {
                reqLowercase.innerHTML = '<span class="tick-icon">❌</span> At least one lowercase letter (a–z)';
            }
            if (/[0-9]/.test(password)) {
                reqNumber.innerHTML = '<span class="tick-icon">✅</span> At least one number (0–9)';
            } else {
                reqNumber.innerHTML = '<span class="tick-icon">❌</span> At least one number (0–9)';
            }
            if (/[!@#$%^&*()_+\-=\[\]{}|;:\'\",.<>\/?]/.test(password)) {
                reqSpecial.innerHTML = '<span class="tick-icon">✅</span> At least one special character (!@#$%^&* etc.)';
            } else {
                reqSpecial.innerHTML = '<span class="tick-icon">❌</span> At least one special character (!@#$%^&* etc.)';
            }
        });
    </script>
</body>
</html>'''

import re
STYLE_BLOCK = "* { margin: 0; padding: 0; box-sizing: border-box; } body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; min-height: 100vh; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); overflow-x: hidden; } .container { max-width: 800px; margin: 0 auto; padding: 20px; min-height: 100vh; } .header { position: relative; text-align: center; margin-bottom: 30px; color: white; } .header h1 { font-size: 3rem; font-weight: 700; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); } .header p { font-size: 1.2rem; opacity: 0.9; } .main-card { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); margin-bottom: 30px; } .journal-section { margin-bottom: 30px; } .journal-section h2 { color: #333; margin-bottom: 15px; font-size: 1.8rem; text-align: center; } .journal-input { width: 100%; max-width: 400px; min-height: 40px; padding: 10px 15px; border: 2px solid #e1e5e9; border-radius: 10px; font-size: 1rem; font-family: inherit; resize: vertical; transition: border-color 0.3s ease; background: rgba(255,255,255,0.9); margin: 0 auto; display: block; } .journal-input:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); } .submit-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 30px; border-radius: 25px; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; display: block; margin: 20px auto; min-width: 200px; } .submit-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3); } .flash-message { padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center; font-weight: 500; } .flash-success { background: linear-gradient(135deg, #43e97b, #38f9d7); color: white; } .flash-error { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }"

if __name__ == '__main__':
    try:
        with app.app_context():
            db.create_all()
            print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Database creation error: {e}")
    
    print("🧠 Dr. Mind is starting up...")
    print("🤖 AI system ready with Hugging Face API!")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    try:
        app.run(debug=debug_mode, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

