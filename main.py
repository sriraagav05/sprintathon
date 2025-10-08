import os
import json
import qrcode
import io
import requests
import base64
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import random

# --- Flask App Initialization ---
load_dotenv()
app = Flask(__name__)

# --- Configuration Setup ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENWEATHER_API_KEY =  '63d95b475932d00be16a35ed303e217b'

# --- FIX: Robust API Key Check ---
# If the key is missing from the environment, assign a placeholder.
if not OPENWEATHER_API_KEY:
    OPENWEATHER_API_KEY = 'MISSING_OPENWEATHER_API_KEY'
    print(
        "--- ⚠️ WARNING: OPENWEATHER_API_KEY is missing from .env. Please set it. Weather features will likely fail. ---")

if not all([app.config['SECRET_KEY'], GOOGLE_API_KEY, OPENWEATHER_API_KEY]):
    print("Warning: Some API keys missing from .env. Functionality may be limited.")
# --- END FIX ---

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'warning'

# --- Configuration & Helpers ---
GROWTH_STAGES = ['Seed', 'Sprout', 'Plant', 'Flowering', 'Harvest']
GROWTH_THRESHOLDS = [0, 10, 25, 45, 70]
POINTS_PER_RUPEE = 1


# --- CHATBOT LOGIC (Updated to handle image upload) ---
def get_chatbot_response(user_message, base64_image=None, mime_type=None, analysis_type=None):
    """
    Gets a conversational response from the Gemini API for the chatbot.
    """
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Using a fast, multimodal model
        model = genai.GenerativeModel(model_name='gemini-2.5-flash-preview-05-20')

        if base64_image:
            print("\n--- CHATBOT: Starting Multimodal Image Analysis ---")
            print(f"MIME Type Received: {mime_type}, Analysis Intent: {analysis_type}")

            try:
                # *** FINAL FIX: Bypass SDK object construction entirely ***
                # Construct the image part as a raw dictionary (the low-level format)
                # The input base64_image is a string, which is correct for this dictionary structure.
                image_part_dict = {
                    'inline_data': {
                        'data': base64_image,
                        'mime_type': mime_type
                    }
                }
                print(
                    f"Image Part dictionary constructed for {analysis_type} intent. Size: {len(base64_image) // 1024} KB.")

            except Exception as construction_error:
                print(f"FATAL CONSTRUCTION ERROR: {construction_error}")
                return "Error: Failed to construct the image request payload."

            # 3. Define the persona and task based on analysis_type
            if analysis_type == 'soil':
                # ADDED CONSTRAINTS: Keep response concise (max 3 points, 4 sentences)
                persona_task = "You are AgroBot, an expert soil scientist. Your task is to analyze the provided image of a soil sample. Describe the soil's texture, color, and likely type (e.g., clay, loam, sandy). Provide recommendations for improving its quality or suitability for common crops. Keep the response concise, using only bullet points for recommendations."
            elif analysis_type == 'disease':
                # ADDED CONSTRAINTS: Keep response concise (max 3 points, 4 sentences)
                persona_task = "You are AgroBot, an expert plant pathologist. Your task is to analyze the provided image of crop leaves/plant parts. **Diagnose the primary issue, and provide a single, clear bulleted list of 3-4 actionable steps for remedy and prevention.** Keep the total diagnosis text brief (max 2 sentences)."
            else:
                persona_task = "You are AgroBot, an expert agricultural analyst. Analyze the provided image of a crop or soil sample. Provide a clear diagnosis or description and suggest actionable next steps. Be concise."

            user_query_text = user_message if user_message else "Please analyze the image based on the requested task."

            # 4. Assemble contents using the dictionary structure directly
            # This structure is necessary when bypassing the SDK's Part object
            contents = [
                f"{persona_task}\n\nUser's Additional Context: {user_query_text}",
                image_part_dict
            ]

            try:
                print("Sending multimodal request to Gemini...")
                # Pass the raw parts list to generate_content
                response = model.generate_content(
                    contents=contents
                )
                print("Multimodal response received successfully.")
                return response.text

            except Exception as e:
                # This block catches API/Network errors specific to the multimodal call
                print(f"Gemini Multimodal API Error: {e}")
                print("--- END CHATBOT Multimodal FAILURE ---")
                return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."


        else:
            # Standard chatbot persona (text-only)
            persona = (
                "You are AgroBot, a friendly and helpful assistant for the AgroVision AI application. "
                "Your purpose is to answer user questions about farming, crops, pest control, and how to use the app's features. "
                "Be concise and provide practical advice, especially relevant for farming in India."
            )
            prompt = f"{persona}\n\nUser's question: {user_message}"
            contents = [prompt]  # Simple text prompt

            response = model.generate_content(
                contents=contents
            )
            return response.text

    except Exception as e:
        # This catches errors specific to the text-only model configuration/call
        print(f"Chatbot API Error (General/Text-only Path): {e}")
        return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."


@app.route('/api/chatbot', methods=['POST'])
@login_required
def chatbot():
    data = request.json
    user_message = data.get('message')
    image_data = data.get('image_data')
    mime_type = data.get('mime_type')
    analysis_type = data.get('analysis_type')

    if not user_message and not image_data:
        return jsonify({"error": "No message or image provided."}), 400

    ai_response = get_chatbot_response(user_message, image_data, mime_type, analysis_type)

    return jsonify({"reply": ai_response})


# --- END CHATBOT LOGIC ---


def gemini_analyze_image(base64_image, crop_name, soil_type):
    """
    Calls the Gemini API using requests to analyze the plant image and return structured JSON.
    NOTE: This function is used by the separate '/api/analyze_plant_photo' route, not the chat.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"

    system_instruction = (
        "You are an agricultural analysis API. Your only function is to return a single, minified JSON object. "
        "Do not output any conversational text, introductions, explanations, or markdown."
    )

    user_prompt = f"""
    Analyze the image of the plant.
    Context: Crop Name: {crop_name}. Soil Type: {soil_type}.
    Constraints: 
    Return ONLY a single JSON object. For the "remedy" and "prevention" fields, provide concise, numbered key points in a single string.
    Example for remedy string: "1. Apply neem oil spray weekly. 
                                2. Remove and destroy heavily infected leaves."
    """

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt},
                                                {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "disease": {"type": "STRING"},
                    "is_curable": {"type": "BOOLEAN"},
                    "remedy": {"type": "STRING"},
                    "prevention": {"type": "STRING"}
                },
                "required": ["disease", "is_curable", "remedy", "prevention"]
            }
        }
    }

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = response.json()
        json_payload = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(json_payload)

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e.response.text if e.response else e}")
        return {"error": "API request failed. Check terminal for details."}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing AI response: {e}")
        return {"error": "Failed to parse a valid response from the AI."}


def create_notification(user_id, message, category='general'):
    notif = Notification(user_id=user_id, message=message, category=category)
    db.session.add(notif)


def calculate_stage_from_days(days):
    days = max(0, days)
    stage = GROWTH_STAGES[0]
    for i, threshold in enumerate(GROWTH_THRESHOLDS):
        if days >= threshold:
            stage = GROWTH_STAGES[i]
    return stage


def calculate_loan_amount(area_acres):
    if area_acres < 5: return 0, "Minimum 5 acres required."
    max_loan = area_acres * 500
    if max_loan > 500000: return 500000, "Max loan is $500,000."
    return max_loan, None


# --- Database Models (Unchanged) ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    points = db.Column(db.Integer, default=0, nullable=False)
    badges_json = db.Column(db.Text, default='[]', nullable=False)
    crops = db.relationship('CropEntry', backref='owner', lazy=True)
    farms = db.relationship('Farm', backref='farmer', lazy=True)
    products_for_sale = db.relationship('Product', backref='seller', lazy=True)
    notifications = db.relationship('Notification', backref='recipient', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_badges(self):
        try:
            return json.loads(self.badges_json)
        except:
            return []

    def add_badge(self, name):
        badges = self.get_badges()
        if name not in badges:
            badges.append(name)
            self.badges_json = json.dumps(badges)

    def __repr__(self):
        return f'<User {self.username}>'


class CropEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    planted_date = db.Column(db.DateTime, default=datetime.utcnow)
    simulated_days = db.Column(db.Integer, default=0)
    soil_type = db.Column(db.String(50), default='Standard')
    disease = db.Column(db.String(100), default='None')
    growth_stage = db.Column(db.String(50), default='Seed')
    status = db.Column(db.String(50), default='Green')
    pesticide_boost_active = db.Column(db.Boolean, default=False)
    last_checked_date = db.Column(db.Date, nullable=True)

    def __repr__(self):
        return f'<CropEntry {self.name}>'


class Farm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    area_acres = db.Column(db.Float, nullable=False)
    name = db.Column(db.String(100), default='My Farm')
    loan_status = db.Column(db.String(50), default='Pending')
    loan_details_json = db.Column(db.Text, default='{}')


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    seller_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    rupee_price = db.Column(db.Float, nullable=False)
    image_url = db.Column(db.String(255), default='https://placehold.co/400x300?text=Image')
    is_available = db.Column(db.Boolean, default=True)
    date_listed = db.Column(db.DateTime, default=datetime.utcnow)

    def get_points_price(self): return int(self.rupee_price * POINTS_PER_RUPEE)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    buyer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    purchase_price = db.Column(db.Integer, nullable=False)
    purchase_date = db.Column(db.DateTime, default=datetime.utcnow)


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    category = db.Column(db.String(50), default='general')


class TodoItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.String(255), nullable=False)
    done = db.Column(db.Boolean, default=False)
    category = db.Column(db.String(50), default='manual')  # 'manual', 'disease_remedy', 'daily'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'done': self.done,
            'timestamp': self.timestamp.isoformat()
        }


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# --- Simulation Logic (Unchanged) ---
def advance_growth(crop, days_increment):
    user = db.session.get(User, crop.user_id)
    if not user:
        return "Error: User not found for this crop.", 500
    if crop.growth_stage == 'Harvest' and days_increment > 0:
        return f"Crop '{crop.name}' has already been harvested!", 200

    rate_multiplier = 1.0
    pesticide_msg = ""
    if days_increment > 0:
        if crop.soil_type == 'Loamy':
            rate_multiplier = 1.2
        elif crop.soil_type == 'Clay':
            rate_multiplier = 0.8
        if crop.pesticide_boost_active:
            rate_multiplier *= 2.0
            crop.pesticide_boost_active = False
            pesticide_msg = "Pesticide boost applied!"

        if crop.disease != 'None':
            crop.status = 'Red'
            db.session.commit()
            return f"Growth halted for '{crop.name}' due to disease: {crop.disease}.", 200

    effective_days = days_increment * rate_multiplier
    new_simulated_days = max(0, crop.simulated_days + effective_days)
    old_stage = crop.growth_stage
    new_stage = calculate_stage_from_days(new_simulated_days)

    has_harvested = (old_stage != 'Harvest') and (new_stage == 'Harvest')

    crop.simulated_days = int(new_simulated_days)
    crop.growth_stage = new_stage

    if crop.disease != 'None':
        crop.status = 'Red'
    elif new_stage != 'Harvest' and GROWTH_THRESHOLDS[-1] > 0 and (new_simulated_days / GROWTH_THRESHOLDS[-1]) > 0.6:
        crop.status = 'Yellow'
    else:
        crop.status = 'Green'

    if has_harvested:
        user.points += 50
        user.add_badge('Green Warrior')
        create_notification(user.id,
                            f"🏆 Harvest Complete! '{crop.name}' reached harvest. You earned 50 points and the 'Green Warrior' badge!",
                            'badge')
        msg = f"Congratulations! '{crop.name}' has reached **Harvest**! You earned 50 points and a new badge."
    elif days_increment < 0:
        msg = f"Pest Attack! '{crop.name}' lost {abs(int(days_increment))} days of growth. Stage is now **{new_stage}**."
    else:
        msg = f"'{crop.name}' advanced by {int(effective_days)} days. Stage: {old_stage} -> **{new_stage}**."

    db.session.commit()
    return f"{msg} {pesticide_msg}".strip(), 200


# --- Frontend Routes (Modified/Added) ---
@app.route('/')
def home():
    if current_user.is_authenticated:
        # Dashboard is the new home page
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/command_center')
@login_required
def command_center():
    """Dedicated route for the Command Center Map Feature (command_center.html)."""
    return render_template('command_center.html')


# Add this utility function near your existing helper functions
def refresh_daily_todos(user_id):
    """Ensures basic, daily To-Do items are present and reset if incomplete."""
    today = date.today()

    # 1. Define the daily task
    daily_task_text = "Water a crop daily (Basic maintenance)"

    # 2. Check for the task based on the text
    daily_task = TodoItem.query.filter_by(
        user_id=user_id,
        text=daily_task_text
    ).order_by(TodoItem.timestamp.desc()).first()

    # 3. If the latest task is complete and from today, or if it doesn't exist, create a new one.
    if not daily_task or daily_task.timestamp.date() < today:
        # If an old, incomplete task exists, mark it as expired/delete it to start fresh.
        if daily_task and daily_task.done == False:
            db.session.delete(daily_task)  # Clean up old incomplete task

        new_daily_todo = TodoItem(user_id=user_id, text=daily_task_text)
        db.session.add(new_daily_todo)
        # Note: We commit this here so it's ready before rendering the dashboard
        db.session.commit()
        return True  # Task was added/refreshed

    return False  # Task is already present and not completed today


# Replace your existing dashboard route with this:
@app.route('/dashboard')
@login_required
def dashboard():
    # --- NEW: Refresh Daily To-Do ---
    refresh_daily_todos(current_user.id)

    # --- (Existing logic to fetch crops_data, leaderboard, etc.) ---
    user_crops = CropEntry.query.filter_by(user_id=current_user.id).order_by(CropEntry.planted_date.desc()).all()
    crops_data = []
    now = datetime.utcnow()
    today_date = date.today()
    for crop in user_crops:
        # ... (Existing crop logic) ...
        time_since_planted = now - crop.planted_date
        real_time_days = time_since_planted.days
        real_time_stage = calculate_stage_from_days(real_time_days)
        is_check_in_available = crop.growth_stage != 'Harvest' and (
                crop.last_checked_date is None or crop.last_checked_date < today_date)
        crops_data.append(
            {'crop': crop, 'real_time_stage': real_time_stage, 'is_check_in_available': is_check_in_available})

    leaderboard = User.query.order_by(User.points.desc()).limit(5).all()
    user_badges = current_user.get_badges()

    # Fetch all To-Do items for rendering (including the newly added daily one)
    todo_items = TodoItem.query.filter_by(user_id=current_user.id).order_by(TodoItem.timestamp.asc()).all()

    return render_template('dashboard.html',
                           crops_data=crops_data,
                           leaderboard=leaderboard,
                           badges=user_badges,
                           todo_items=todo_items
                           )


@app.route('/scan')
@login_required
def scan():
    user_crops = CropEntry.query.filter_by(user_id=current_user.id).all()
    return render_template('scan.html', crops=user_crops)


@app.route('/simulate_page', defaults={'crop_id': None})
@app.route('/simulate_page/<int:crop_id>')
@login_required
def simulate_page(crop_id):
    user_crops = CropEntry.query.filter_by(user_id=current_user.id).all()
    return render_template('simulate.html', crops=user_crops, pre_select_id=crop_id)


@app.route('/loan_simulator')
@login_required
def loan_simulator():
    user_farms = Farm.query.filter_by(user_id=current_user.id).all()
    return render_template('loan_simulator.html', farms=user_farms)


@app.route('/shop')
@login_required
def shop():
    current_user_points = current_user.points
    available_products = Product.query.filter(
        Product.seller_id != current_user.id,
        Product.is_available == True
    ).all()
    purchase_history = db.session.query(Transaction, Product.name).join(Product).filter(
        Transaction.buyer_id == current_user.id
    ).order_by(Transaction.purchase_date.desc()).all()
    user_listings = Product.query.filter_by(seller_id=current_user.id).all()
    return render_template(
        'shop.html',
        products=available_products,
        history=purchase_history,
        user_listings=user_listings,
        current_user_points=current_user_points
    )


@app.route('/notifications')
@login_required
def notifications_page():
    user_notifications = current_user.notifications.order_by(Notification.timestamp.desc()).limit(10).all()
    for notif in user_notifications:
        notif.is_read = True
    db.session.commit()
    return render_template('notifications.html', notifications=user_notifications)


# --- API Routes ---

# 1. To-Do List CRUD API Routes (Consolidated for robust registration)
@app.route('/api/todo', methods=['POST'])
@login_required
def add_todo():
    data = request.get_json()
    new_task_text = data.get('text', '').strip()

    if not new_task_text:
        return jsonify({'message': 'Task text cannot be empty'}), 400

    new_item = TodoItem(user_id=current_user.id, text=new_task_text)
    db.session.add(new_item)
    db.session.commit()

    return jsonify(new_item.to_dict()), 201


@app.route('/api/todo/<int:todo_id>', methods=['DELETE'])
@login_required
def delete_todo(todo_id):
    item = db.session.get(TodoItem, todo_id)

    if not item or item.user_id != current_user.id:
        return jsonify({'message': 'To-Do item not found or unauthorized'}), 404

    db.session.delete(item)
    db.session.commit()

    return jsonify({'message': 'To-Do item deleted', 'id': todo_id}), 200


@app.route('/api/todo/toggle/<int:todo_id>', methods=['POST'])
@login_required
def toggle_todo(todo_id):
    item = db.session.get(TodoItem, todo_id)

    if not item or item.user_id != current_user.id:
        return jsonify({'message': 'To-Do item not found or unauthorized'}), 404

    # Check if the task is being marked as DONE (not undone)
    if not item.done:
        # Daily Task Reward Logic
        if "Water a crop daily" in item.text:
            current_user.points += 2  # Reward small amount for basic tasks
            new_points = current_user.points
            flash(f"Daily task complete! +2 Points.", 'success')

        # Disease Task Reward Logic (Example: larger reward)
        elif "Apply pesticide/remedy" in item.text:
            current_user.points += 15  # Reward larger amount for critical tasks
            new_points = current_user.points
            flash(f"Disease remedy applied! +15 Points.", 'success')

        else:
            new_points = current_user.points

        item.done = True
        db.session.commit()
        return jsonify({
            'message': 'To-Do item completed!',
            'id': todo_id,
            'done': True,
            'new_points': new_points  # Return points for UI update
        }), 200

    # If the user is un-checking a task, just toggle it back without penalizing points
    item.done = False
    db.session.commit()
    return jsonify({
        'message': 'To-Do item untoggled',
        'id': todo_id,
        'done': False,
        'new_points': current_user.points
    }), 200


# 2. Existing Weather and other API routes

@app.route('/api/get_current_weather')
@login_required
def get_current_weather():
    """Simple API for the map banner (only needs current conditions)."""
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required"}), 400

    # --- FIX: Use robust error handling for API Key issues ---
    if OPENWEATHER_API_KEY == 'MISSING_OPENWEATHER_API_KEY':
        return jsonify({"error": "OpenWeather API Key is missing. Check your .env file."}), 500

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()  # Check for 4xx or 5xx status codes
        data = response.json()

        weather = {
            "temp": round(data['main']['temp']),
            "description": data['weather'][0]['description'],
            "id": data['weather'][0]['id']  # The weather condition code
        }
        return jsonify(weather)
    except requests.exceptions.HTTPError as http_err:
        # This catches errors like 401 (Unauthorized - common API key failure)
        print(f"Weather API HTTP Error (Status {response.status_code}): {http_err}")
        return jsonify({
                           "error": f"Weather API error (Status {response.status_code}). Check your OpenWeather API Key."}), response.status_code
    except Exception as e:
        print(f"Weather API general error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/weather')
@login_required
def get_weather():
    """Complex API for irrigation planner (needs area and forecast)."""
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    area_acres_str = request.args.get('area')

    if not all([lat, lon, area_acres_str]):
        return jsonify({"error": "Latitude and Longitude are required"}), 400

    # --- FIX: Use robust error handling for API Key issues ---
    if OPENWEATHER_API_KEY == 'MISSING_OPENWEATHER_API_KEY':
        return jsonify({"error": "OpenWeather API Key is missing. Check your .env file."}), 500

    try:
        area_acres = float(area_acres_str)
    except ValueError:
        return jsonify({"error": "Invalid area format"}), 400

    try:
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"

        current_response = requests.get(current_url)
        current_response.raise_for_status()
        forecast_response = requests.get(forecast_url)
        forecast_response.raise_for_status()

        current_data = current_response.json()
        forecast_data = forecast_response.json()

        if current_data.get('cod') != 200 or forecast_data.get('cod') != "200":
            return jsonify({"error": "Failed to retrieve complete weather data"}), 400

        # --- IRRIGATION PLANNER LOGIC ---
        irrigation_plan = []
        daily_data = {}

        for entry in forecast_data.get('list', []):
            day = datetime.fromtimestamp(entry['dt']).strftime('%Y-%m-%d')
            if day not in daily_data:
                daily_data[day] = []
            daily_data[day].append(entry)

        for day, entries in list(daily_data.items())[:5]:
            day_temps = [e['main']['temp'] for e in entries]
            total_rain = sum(e.get('rain', {}).get('3h', 0) for e in entries)
            avg_temp = sum(day_temps) / len(day_temps)

            irrigation_hours = 1.5 * area_acres

            if avg_temp > 32:
                irrigation_hours *= 1.25

            if total_rain > 5:
                recommendation = "No Irrigation Needed"
            else:
                recommendation = f"Irrigate: {irrigation_hours:.1f} hours"

            irrigation_plan.append({
                "day_name": datetime.strptime(day, '%Y-%m-%d').strftime('%A'),
                "icon": entries[0]['weather'][0]['icon'],
                "avg_temp": round(avg_temp),
                "recommendation": recommendation
            })
        # --- END OF IRRIGATION LOGIC ---

        clean_data = {
            "temperature": round(current_data.get('main', {}).get('temp', 0)),
            "description": current_data.get('weather', [{}])[0].get('description', 'N/A'),
            "icon": current_data.get('weather', [{}])[0].get('icon', '01d'),
            "irrigation_plan": irrigation_plan
        }
        return jsonify(clean_data)

    except requests.exceptions.HTTPError as http_err:
        print(
            f"Weather API HTTP Error (Status {current_response.status_code if 'current_response' in locals() else forecast_response.status_code}): {http_err}")
        return jsonify({
                           "error": f"Weather API error (Status {current_response.status_code if 'current_response' in locals() else forecast_response.status_code}). Check your OpenWeather API Key."}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to process weather data"}), 500


@app.route('/api/get_land_report')
@login_required
def get_land_report():
    """Simulates a detailed land suitability and valuation report for the map panel."""
    # --- Data Simulation (Soil, Water, Land Value, etc.) ---
    lat_str = request.args.get('lat', '0')
    lat = float(lat_str)
    soil_types = ["Alluvial Soil", "Clay Loam", "Red Loam", "Sandy Soil"]
    soil_type = random.choice(soil_types)
    nitrogen = round(random.uniform(0.5, 2.5), 2)
    phosphorus = round(random.uniform(10, 150), 1)
    potassium = round(random.uniform(50, 400), 1)
    groundwater_depth = random.randint(30, 300)
    yearly_rainfall_mm = [random.randint(20, 100) for _ in range(12)]
    ten_year_avg_mm = [random.randint(40, 80) for _ in range(12)]
    base_price = random.randint(400000, 1500000)
    past_price = int(base_price * random.uniform(0.6, 0.8))
    future_price = int(base_price * random.uniform(1.2, 1.5))
    is_wetter_than_average = sum(yearly_rainfall_mm) > sum(ten_year_avg_mm)

    # --- Expert Analysis Logic ---
    if nitrogen < 1.0 or phosphorus < 30 or potassium < 100:
        nutrient_level = 'poor'
    elif nitrogen > 2.0 or phosphorus > 120 or potassium > 350:
        nutrient_level = 'good'
    else:
        nutrient_level = 'average'
    nutrient_text = "Nutrient levels are strong. Optimal for heavy feeders." if nutrient_level == 'good' else "Key nutrient levels are low; we recommend soil testing and organic fertilizer application." if nutrient_level == 'poor' else "Nutrient levels are adequate, suitable for most common crops."

    if groundwater_depth > 200:
        water_level = 'poor'
    elif groundwater_depth < 75:
        water_level = 'good'
    else:
        water_level = 'average'
    water_text = f"Groundwater is deep at {groundwater_depth} ft (pumping costs may be higher; consider drip irrigation)." if water_level == 'poor' else f"Groundwater is shallow at {groundwater_depth} ft (excellent water accessibility for flood or pivot irrigation)." if water_level == 'good' else f"Groundwater is at a moderate depth of {groundwater_depth} ft."

    if sum(yearly_rainfall_mm) > sum(ten_year_avg_mm) * 1.2:
        rainfall_summary = "Last year was significantly wetter than average, suggesting higher residual soil moisture."
    elif sum(yearly_rainfall_mm) < sum(ten_year_avg_mm) * 0.8:
        rainfall_summary = "Last year was drier than average, indicating potential drought stress for the current season."
    else:
        rainfall_summary = "Rainfall was consistent with the long-term average, providing reliable water input."

    # --- Climate Resilience Logic ---
    base_risk = abs(lat) / 90.0
    random_factor = random.uniform(-0.2, 0.2)
    final_risk_score = max(0, min(1, base_risk + random_factor))

    if final_risk_score > 0.7:
        resilience_rating, resilience_level, temp_increase, drought_risk, climate_recommendation = "C: High Risk", "poor", round(
            random.uniform(2.0, 3.0),
            1), "High", "Focus on highly drought-resistant crops and water harvesting techniques (e.g., check dams)."
    elif final_risk_score > 0.4:
        resilience_rating, resilience_level, temp_increase, drought_risk, climate_recommendation = "B: Moderate Risk", "average", round(
            random.uniform(1.5, 2.5),
            1), "Moderate", "Invest in efficient irrigation systems (drip/sprinkler) and resilient crop varieties."
    else:
        resilience_rating, resilience_level, temp_increase, drought_risk, climate_recommendation = "A: Low Risk", "good", round(
            random.uniform(1.0, 2.0),
            1), "Low", "Land is relatively resilient. Continue standard best practices and monitor long-term trends."

    # --- AI Crop Matchmaker Logic ---
    CROP_PROFILES = {
        "Rice": {"soil": ["Alluvial Soil", "Clay Loam"], "water": "good", "rainfall": True},
        "Sugarcane": {"soil": ["Clay Loam", "Red Loam"], "water": "average", "rainfall": True},
        "Cotton": {"soil": ["Alluvial Soil", "Red Loam"], "water": "poor", "rainfall": False},
        "Groundnut": {"soil": ["Sandy Soil", "Red Loam"], "water": "average", "rainfall": False}
    }
    crop_suitability = []
    for crop, profile in CROP_PROFILES.items():
        score = 100
        if soil_type not in profile["soil"]: score -= 35
        if water_level != profile["water"]: score -= 25
        if is_wetter_than_average != profile["rainfall"]: score -= 20
        crop_suitability.append({"name": crop, "score": max(20, score) + random.randint(-5, 5)})
    crop_suitability.sort(key=lambda x: x["score"], reverse=True)

    # --- Final Report Assembly ---
    score = 0
    if nutrient_level == 'good': score += 1
    if water_level == 'good': score += 1
    if resilience_level == 'good': score += 1

    if score >= 2.5:
        overall_rating = "Excellent Investment"
    elif score >= 1.5:
        overall_rating = "Good for Cultivation"
    else:
        overall_rating = "Challenging Conditions"

    report = {
        "overall_rating": overall_rating,
        "analysis": {
            "nutrients": {"level": nutrient_level, "text": nutrient_text},
            "water": {"level": water_level, "text": water_text}
        },
        "climate_resilience": {
            "rating": resilience_rating,
            "level": resilience_level,
            "projections": {
                "temp_increase_2040": temp_increase,
                "drought_risk": drought_risk
            },
            "recommendation": climate_recommendation
        },
        "crop_suitability": crop_suitability,
        "land_value": {"past": past_price, "present": base_price, "future": future_price},
        "historical_rainfall": {
            "summary": rainfall_summary,
            "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            "last_12_months": yearly_rainfall_mm,
            "ten_year_average": ten_year_avg_mm
        },
        "soil_type": soil_type
    }
    return jsonify(report)


@app.route('/api/analyze_plant_photo', methods=['POST'])
@login_required
def analyze_plant_photo():
    if 'plant_photo' not in request.files: return jsonify({"error": "No file part."}), 400
    photo_file = request.files['plant_photo']
    crop_id = request.form.get('crop_id')
    soil_type = request.form.get('soil_type')
    if not all([photo_file.filename, crop_id, soil_type]): return jsonify({"error": "Missing form data."}), 400

    crop = db.session.get(CropEntry, int(crop_id))
    if not crop or crop.user_id != current_user.id: return jsonify({"error": "Crop not found or unauthorized."}), 404

    image_bytes = photo_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    analysis_result = gemini_analyze_image(base64_image, crop.name, soil_type)

    if "error" in analysis_result:
        return jsonify({"error": f"AI Analysis Failed: {analysis_result['error']}"}), 500

    return jsonify(analysis_result), 200


@app.route('/create_crop', methods=['POST'])
@login_required
def create_crop():
    crop_name = request.form.get('crop_name')
    initial_stage = request.form.get('initial_stage', 'Seed')
    if not crop_name:
        flash('Crop name is required.', 'danger')
        return redirect(url_for('dashboard'))
    new_crop = CropEntry(user_id=current_user.id, name=crop_name, growth_stage=initial_stage)
    db.session.add(new_crop)
    db.session.commit()
    flash(f"New crop '{crop_name}' planted!", 'success')
    return redirect(url_for('dashboard'))


@app.route('/create_farm', methods=['POST'])
@login_required
def create_farm():
    try:
        farm_name = request.form.get('farm_name', 'My Farm')
        area_acres = float(request.form.get('area_acres'))
        if area_acres <= 0: raise ValueError()
    except (ValueError, TypeError):
        flash('Invalid area provided. Must be a positive number.', 'danger')
        return redirect(url_for('loan_simulator'))
    new_farm = Farm(user_id=current_user.id, name=farm_name, area_acres=area_acres)
    db.session.add(new_farm)
    db.session.commit()
    flash(f"Farm '{farm_name}' added.", 'success')
    return redirect(url_for('loan_simulator'))


@app.route('/create_product', methods=['POST'])
@login_required
def create_product():
    name = request.form.get('name', '').strip()
    rupee_price_str = request.form.get('rupee_price', '').strip()

    if not name:
        flash('Product name is required.', 'danger')
        return redirect(url_for('shop'))

    try:
        rupee_price = float(rupee_price_str)
        if rupee_price <= 0:
            raise ValueError()
    except ValueError:
        flash('Product price is required and must be a positive number.', 'danger')
        return redirect(url_for('shop'))

    description = request.form.get('description', '').strip()
    image_url = request.form.get('image_url', '').strip()
    if not image_url or not image_url.startswith(('http', 'https')):
        image_url = 'https://placehold.co/400x300?text=Image'

    new_product = Product(
        seller_id=current_user.id,
        name=name,
        description=description,
        rupee_price=rupee_price,
        image_url=image_url
    )
    db.session.add(new_product)
    db.session.commit()
    flash(f"Product '{name}' listed successfully!", 'success')
    return redirect(url_for('shop'))


@app.route('/api/underwrite_loan/<int:id>', methods=['POST'])
@login_required
def underwrite_loan(id):
    farm = db.session.get(Farm, id)
    if not farm: return jsonify({"message": "Farm not found"}), 404
    if farm.user_id != current_user.id: return jsonify({"message": "Unauthorized"}), 403

    max_loan, reason = calculate_loan_amount(farm.area_acres)
    details = {"area": farm.area_acres, "max_loan_amount": max_loan, "date_analyzed": datetime.utcnow().isoformat()}

    if reason:
        farm.loan_status = 'Denied'
        details['denial_reason'] = reason
        message = f"Loan Denied for {farm.name}: {reason}"
        status = 400
    else:
        farm.loan_status = 'Approved'
        message = f"Loan Approved for {farm.name}! Max loan: ${max_loan:,.2f}."
        status = 200

    farm.loan_details_json = json.dumps(details)
    db.session.commit()
    return jsonify({"message": message, "status": farm.loan_status, "loan_details": details}), status


@app.route('/api/buy_product/<int:id>', methods=['POST'])
@login_required
def buy_product(id):
    product = db.session.get(Product, id)
    if not product: return jsonify({"message": "Product not found."}), 404
    if product.seller_id == current_user.id: return jsonify({"message": "You cannot buy your own product."}), 400
    if not product.is_available: return jsonify({"message": f"'{product.name}' is no longer available."}), 400

    price_in_points = product.get_points_price()
    if current_user.points < price_in_points:
        return jsonify({"message": f"Insufficient points. You need {price_in_points}."}), 400

    seller = db.session.get(User, product.seller_id)
    current_user.points -= price_in_points
    seller.points += price_in_points
    product.is_available = False

    new_transaction = Transaction(product_id=product.id, buyer_id=current_user.id, purchase_price=price_in_points)
    db.session.add(new_transaction)

    create_notification(seller.id,
                        f"💰 SOLD! '{product.name}' was bought by {current_user.username} for {price_in_points} points.",
                        'sale')
    db.session.commit()

    flash(f"Successfully bought '{product.name}'!", 'success')
    return jsonify({"message": "Purchase complete!", "new_points": current_user.points}), 200


@app.route('/api/check_in/<int:id>', methods=['POST'])
@login_required
def daily_check_in(id):
    crop = db.session.get(CropEntry, id)
    if not crop: return jsonify({"message": "Crop not found."}), 404
    if crop.user_id != current_user.id: return jsonify({"message": "Unauthorized"}), 403
    if crop.growth_stage == 'Harvest': return jsonify({"message": "Cannot check in on a harvested crop."}), 400

    today_date = date.today()
    if crop.last_checked_date == today_date: return jsonify({"message": "Already checked in today."}), 200

    current_user.points += 5
    crop.last_checked_date = today_date
    db.session.commit()

    flash(f"Daily Check-in for '{crop.name}' successful! +5 Points.", 'success')
    return jsonify({"message": f"Check-in successful! +5 Points.", "new_points": current_user.points}), 200


@app.route('/qr_code/<int:id>')
@login_required
def qr_code(id):
    crop = db.session.get(CropEntry, id)
    if not crop: return "Not Found", 404
    if crop.user_id != current_user.id:
        return "Unauthorized", 403
    qr_data = f"AgroVision AI Crop ID: {crop.id}, User: {crop.user_id}"
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=5, border=4)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="green", back_color="white").convert('RGB')
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', as_attachment=False)


@app.route('/simulate/<int:id>', methods=['POST'])
@login_required
def simulate_crop(id):
    crop = db.session.get(CropEntry, id)
    if not crop: return jsonify({"message": "Not Found"}), 404
    if crop.user_id != current_user.id:
        return jsonify({"message": "Unauthorized"}), 403
    try:
        days_increment = int(request.form.get('days_increment', 0))
    except ValueError:
        return jsonify({"message": "Invalid days increment."}), 400
    if not 0 < days_increment <= 10:
        return jsonify({"message": "Increment must be between 1 and 10."}), 400
    message, status_code = advance_growth(crop, days_increment)
    return jsonify({"message": message, "new_stage": crop.growth_stage, "new_days": crop.simulated_days}), status_code


@app.route('/api/simulate_setback/<int:id>', methods=['POST'])
@login_required
def simulate_setback(id):
    crop = db.session.get(CropEntry, id)
    if not crop: return jsonify({"message": "Not Found"}), 404
    if crop.user_id != current_user.id:
        return jsonify({"message": "Unauthorized"}), 403
    if crop.growth_stage == 'Harvest':
        return jsonify({"message": "Cannot apply setback to harvested crop."}), 400
    message, status_code = advance_growth(crop, -5)
    return jsonify({"message": message, "new_stage": crop.growth_stage, "new_days": crop.simulated_days}), status_code


@app.route('/api/apply_pesticide/<int:id>', methods=['POST'])
@login_required
def apply_pesticide(id):
    crop = db.session.get(CropEntry, id)
    if not crop: return {"error": "Not Found"}, 404
    if crop.user_id != current_user.id:
        return {"error": "Unauthorized"}, 403
    if crop.growth_stage == 'Harvest':
        return {"error": "Cannot apply boost to harvested crop."}, 400
    if crop.pesticide_boost_active:
        flash(f"Pesticide boost already queued for '{crop.name}'.", 'warning')
        return {"message": "Pesticide boost already queued."}, 200
    crop.pesticide_boost_active = True
    db.session.commit()
    flash(f"Pesticide boost queued for '{crop.name}'.", 'info')
    return {"message": "Pesticide boost queued."}, 200


@app.route('/api/save_scan_result', methods=['POST'])
@login_required
def save_scan_result():
    data = request.json
    crop_id = data.get('crop_id')
    soil_type = data.get('soil_type')
    disease = data.get('disease')

    crop = db.session.get(CropEntry, int(crop_id))
    if not crop or crop.user_id != current_user.id: return {"error": "Crop not found or unauthorized."}, 404

    if soil_type in ['Loamy', 'Clay', 'Standard']: crop.soil_type = soil_type

    if disease and disease != 'None':
        # --- NEW LOGIC: DISEASE DETECTED ---
        crop.disease = disease
        crop.status = 'Red'

        # 1. Add 'Apply Pesticide' Task (if not already present for this crop)
        pesticide_task_text = f"Apply pesticide/remedy to {crop.name} for {disease}"

        # Check if task already exists
        exists = TodoItem.query.filter_by(user_id=current_user.id, text=pesticide_task_text, done=False).first()
        if not exists:
            new_todo = TodoItem(user_id=current_user.id, text=pesticide_task_text)
            db.session.add(new_todo)
            flash(f"Scan detected **{disease}** on '{crop.name}'. An urgent To-Do task has been created.", 'danger')
        else:
            flash(f"Scan detected **{disease}** on '{crop.name}'. Status is RED.", 'danger')

    else:
        # --- NEW LOGIC: DISEASE RESOLVED ---
        crop.disease = 'None'
        if crop.growth_stage != 'Harvest':
            crop.status = 'Green'

        # 2. Automatically remove pending disease tasks for this crop
        pesticide_task_text_prefix = f"Apply pesticide/remedy to {crop.name}"
        TodoItem.query.filter(
            TodoItem.user_id == current_user.id,
            TodoItem.text.startswith(pesticide_task_text_prefix),
            TodoItem.done == False
        ).delete(synchronize_session=False)

        flash(f"Scan detected no diseases on '{crop.name}'. Status is GREEN.", 'success')

    db.session.commit()
    return {"message": "Scan results saved.", "crop_name": crop.name}, 200


# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Check username and password.', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# --- Main Run Block ---
if __name__ == '__main__':
    with app.app_context():
        db_path = os.path.join(app.instance_path, 'site.db')
        if not os.path.exists(db_path):
            db.create_all()
            print("Database initialized.")
    app.run(debug=True)
