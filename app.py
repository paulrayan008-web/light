import os
import requests
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from flask import Flask, render_template, request, redirect, url_for, session,flash
from torchvision import transforms
import torch.optim as optim
from torchvision import models, transforms
from werkzeug.exceptions import RequestEntityTooLarge
import psycopg2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import psycopg2.extras

# =============================
# CONFIGURATION
# =============================

pytesseract_path = r"C:\Users\rayan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 500MB ALLOWED

app.secret_key = "streetlight_secret_key"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 # ✅ Store this in DB
# app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB ALLOWED


API_KEY = "dbe2bec1-0dbf-11f1-bcb0-0200cd936042"

# =============================
# DATABASE CONNECTION
# =============================

def get_db_connection():
    return psycopg2.connect(
        host="aws-1-ap-south-1.pooler.supabase.com",
        database="postgres",
        user="postgres.ohtpdxrtodcdjevqwujd",
        password="Qi6HqIeoJV7NTJ3R",
        port="5432",
        sslmode="require"
    )

# @app.errorhaaseityTooLarge)
# def handle_large_file(e):
#     return """
#     <script>
#         alert("File Too Large! Please upload image below 20MB.");
#         window.history.back();
#     </script>
#     """, 413

# =============================
# CNN MODEL
# =============================

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.mobilenet_v2(pretrained=True)
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Modify classifier
model.classifier[1] = nn.Linear(model.last_channel, 2)

model = model.to(device)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, 4)
        # )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ✅ Load NEW MODEL
model = CNN().to(device)
model.load_state_dict(torch.load("streetlight_multiclass.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ⚠ MUST MATCH training order
class_names = ["low", "Light_of", "Light_on", "physical"]

# =============================
# ROUTES
# =============================

@app.route('/')
def home():
    return render_template('user_login.html')
@app.route('/about')
def about():
    return render_template('about.html')

# -----------------------------
# USER LOGIN WITH OTP
# -----------------------------
@app.route('/user', methods=['GET', 'POST'])
def user():
    phone = session.get('verified_phone')
    otp_sent = False
    error = None

    if request.method == 'POST':
        action = request.form.get('action')

        if action == "send_otp":
            phone = request.form.get('phone')
            if phone and phone.isdigit() and len(phone) == 10:
                url = f"https://2factor.in/API/V1/{API_KEY}/SMS/{phone}/AUTOGEN/SLFD_OTP"
                response = requests.get(url)
                data = response.json()
                if data["Status"] == "Success":
                    session['phone'] = phone
                    session['session_id'] = data["Details"]
                    otp_sent = True
                else:
                    error = "Failed to send OTP"
            else:
                error = "Enter valid 10 digit number"

        elif action == "verify_otp":
            entered_otp = request.form.get('entered_otp')
            session_id = session.get('session_id')
            if session_id:
                url = f"https://2factor.in/API/V1/{API_KEY}/SMS/VERIFY/{session_id}/{entered_otp}"
                response = requests.get(url)
                data = response.json()
                if data["Status"] == "Success":
                    session['verified_phone'] = session.get('phone')
                    return redirect(url_for('predict_analysis'))
                else:
                    otp_sent = True
                    phone = session.get('phone')
                    error = "Invalid OTP"
            else:
                error = "Session expired. Try again."

    return render_template("login.html", otp_sent=otp_sent, error=error)

# -----------------------------
# PREDICT / COMPLAINT PAGE
# -----------------------------
@app.route('/predict_analysis', methods=['GET', 'POST'])
def predict_analysis():

    if "verified_phone" not in session:
        return redirect("/")

    phone = session.get("verified_phone")

    # =========================
    # FIRST: If GET → show page
    # =========================
    if request.method == "GET":
        return render_template("complaint.html")

    # =========================
    # ONLY POST BELOW
    # =========================

    post_id = request.form.get("post_id")

    if not post_id:
        return "<script>alert('Enter Post ID');window.location='/predict_analysis';</script>"

    db = get_db_connection()
    cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # Fetch post details
    cursor.execute("SELECT * FROM post WHERE post_id=%s", (post_id,))
    post_data = cursor.fetchone()

    if not post_data:
        db.close()
        return "<script>alert('Invalid Post ID');window.location='/predict_analysis';</script>"

    area = post_data["area"]
    employee_name = post_data["employee_name"]

    # ======================================
    # IMAGE HANDLING
    # ======================================
    import base64
    from io import BytesIO

    captured_image = request.form.get("captured_image")
    image_file = request.files.get("image")

    if captured_image and captured_image.strip() != "":
        image_data = captured_image.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
        filename = post_id + "_camera.png"

    elif image_file and image_file.filename != "":
        image = Image.open(image_file).convert("RGB")
        filename = post_id + "_" + image_file.filename

    else:
        db.close()
        return "<script>alert('Capture or Upload Image');window.location='/predict_analysis';</script>"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(filepath)

    # ======================================
    # CNN PREDICTION
    # ======================================
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    fault = class_names[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)
    if confidence <0:
        return render_template(
            "complaint.html",
              error="Invalid Image! Please upload a clear streetlight image."
            )

    # print("Prediction:", fault)
    # print("Confidence:", confidence_score, "%")
    

    # ======================================
    # USER INPUT
    # ======================================
    fault1 = request.form.get("fault1")
    fault2 = request.form.get("fault2")
    fault3 = request.form.get("fault3")
    suggestion = request.form.get("suggestion")

    # ======================================
    # INSERT INTO DATABASE
    # ======================================
    cursor.execute("""
        INSERT INTO complaints
        (phone, post_id, area, employee_name, cnn_result, confidence,
         fault1, fault2, fault3, suggestion, image_path, status)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)    RETURNING id
    """  , (
        phone, post_id, area, employee_name,
        fault, confidence_score,
        fault1, fault2, fault3,
        suggestion,
        "uploads/" + filename,   # ✅ FIXED IMAGE PATH
        "Pending"
    ))

    db.commit()
    complaint_id = cursor.fetchone()[0]  # this gets the id column value

    db.close()
   

    return render_template(
        "complaint.html",
        success="Complaint Submitted Successfully!",
        fault=fault,
        confidence=confidence_score,
        area=area,
        employee_name=employee_name,
        image_path=url_for("static", filename="uploads/" + filename),
        id=complaint_id   # ✅ PASS id

    )

# -----------------------------
# EMPLOYEE LOGIN / REGISTER / DASHBOARD
# -----------------------------
@app.route('/employee_login', methods=['GET','POST'])
def employee_login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        db = get_db_connection()
        cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT * FROM employees WHERE email=%s AND password=%s",
                       (email,password))
        employee = cursor.fetchone()
        db.close()

        if employee:
            session["employee_id"] = employee["id"]
            session["employee_name"] = employee["name"]
            session["employee_area"] = employee["area"]
            return redirect(url_for("employee_dashboard"))
        else:
            return "<script>alert('Invalid Login');window.location='/employee_login';</script>"

    return render_template("employeelogin.html")
#employee register
@app.route('/employee_register', methods=['GET', 'POST'])
def employee_register():

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        area = request.form['area']
        password = request.form['password']

        db = get_db_connection()
        cursor = db.cursor()

        # check if email already exists
        cursor.execute("SELECT * FROM employees WHERE email=%s", (email,))
        existing = cursor.fetchone()

        if existing:
            db.close()
            return "<script>alert('Email already registered');window.location='/employee_register';</script>"

        cursor.execute("""
            INSERT INTO employees (name,email,phone,area,password)
            VALUES (%s,%s,%s,%s,%s)
        """, (name,email,phone,area,password))

        db.commit()
        db.close()

        return "<script>alert('Registration Successful');window.location='/employee_login';</script>"

    return render_template('employeereg.html')
#employee dashboard
@app.route('/employee_dashboard')
def employee_dashboard():

    if "employee_id" not in session:
        return redirect("/employee_login")

    db = get_db_connection()
    cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cursor.execute("""
        SELECT * FROM complaints
        WHERE area=%s AND status!='Resolved'
    """, (session["employee_area"],))

    complaints = cursor.fetchall()
    db.close()

    return render_template("dashemp.html",
                           complaints=complaints,
                           area=session["employee_area"])
#resolved

@app.route('/mark_resolved/<int:id>')
def mark_resolved(id):

    db = get_db_connection()
    cursor = db.cursor()

    cursor.execute("""
        UPDATE complaints
        SET status='Resolved'
        WHERE id=%s
    """, (id,))

    db.commit()
    db.close()

    return redirect("/employee_dashboard")

# -----------------------------
# ADMIN LOGIN / DASHBOARD
# -----------------------------
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Default credentials (NO DATABASE)
        if username == "admin" and password == "admin123":
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid Username or Password!")

    return render_template('admin_login.html')
#dashboard

@app.route('/admin_dashboard')
def admin_dashboard():
    db = get_db_connection()
    cursor = db.cursor()

    # Total Complaints
    cursor.execute("SELECT COUNT(*) FROM complaints")
    total_complaints = cursor.fetchone()[0]

    # Pending Complaints
    cursor.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Pending'")
    pending_complaints = cursor.fetchone()[0]

    # Resolved Complaints
    cursor.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Resolved'")
    resolved_complaints = cursor.fetchone()[0]

    # Total Employees
    cursor.execute("SELECT COUNT(*) FROM employees")
    total_employees = cursor.fetchone()[0]

    cursor.close()
    db.close()

    return render_template(
        'dashboard.html',
        total_complaints=total_complaints,
        pending_complaints=pending_complaints,
        resolved_complaints=resolved_complaints,
        total_employees=total_employees
    )

#admin complaints 
@app.route('/admin_complaints')
def admin_complaints():

    db = get_db_connection()
    cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cursor.execute("SELECT * FROM complaints")
    complaints = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template("admin-complaints.html", complaints=complaints)
#updte
@app.route('/update_complaint/<int:id>', methods=['POST'])
def update_complaint(id):
    employee_name = request.form['employee_name']
    status = request.form['status']

    
    db = get_db_connection()
    cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cursor.execute("""
        UPDATE complaints 
        SET employee_name=%s, status=%s 
        WHERE id=%s
    """, (employee_name, status, id))

    db.commit()
    cursor.close()
    db.close()

    return redirect(url_for('admin_complaints'))
#delete
@app.route('/delete_complaint/<int:id>')
def delete_complaint(id):

   
    db = get_db_connection()
    cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cursor.execute("DELETE FROM complaints WHERE id=%s", (id,))
    db.commit()

    cursor.close()
    db.close()

    return redirect(url_for('admin_complaints'))
#employee management
@app.route('/admin_employees')
def admin_employees():
    db = get_db_connection()
    cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()
    db.close()
    return render_template('admin-employees.html', employees=employees)
# ---------------- VIEW ----------------
@app.route("/view_employees")
def view_employees():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template("view_employees.html", employees=employees)

# ---------------- DELETE ----------------
@app.route("/delete/<int:id>")
def delete_employee(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM employees WHERE id=%s", (id,))
    conn.commit()
    cursor.close()
    conn.close()

    return redirect(url_for("view_employees"))

# ---------------- EDIT ----------------
@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit_employee(id):
    conn = get_db_connection()
    cursor = conn.cursor()

    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        area = request.form["area"]

        cursor.execute("""
            UPDATE employees 
            SET name=%s, email=%s, phone=%s, area=%s 
            WHERE id=%s
        """, (name, email, phone, area, id))

        conn.commit()
        cursor.close()
        conn.close()

        return redirect(url_for("view_employees"))

    cursor.execute("SELECT * FROM employees WHERE id=%s", (id,))
    employee = cursor.fetchone()
    cursor.close()
    conn.close()

    return render_template("edit_employee.html", emp=employee)
#report
from flask import send_file
@app.route('/generate_report/<int:id>')
def generate_report(id):

    db = get_db_connection()
    cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cursor.execute("SELECT * FROM complaints WHERE id=%s", (id,))
    complaint = cursor.fetchone()

    cursor.close()
    db.close()

    if complaint is None:
        return f"Complaint ID {id} Not Found"

    # continue PDF code...

    # Create reports folder
    reports_folder = os.path.join(app.root_path, "static", "reports")
    os.makedirs(reports_folder, exist_ok=True)

    pdf_filename = f"complaint_{id}.pdf"
    pdf_path = os.path.join(reports_folder, pdf_filename)

    # Create PDF
    doc = SimpleDocTemplate(pdf_path)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("Street Light Complaint Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.5 * inch))

    # Complaint Details
    elements.append(Paragraph(f"<b>Complaint ID:</b> {complaint.get('id', '')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Phone:</b> {complaint.get('phone', '')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Post ID:</b> {complaint.get('post_id', '')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Area:</b> {complaint.get('area', '')}", styles["Normal"]))

    # User Selected Fault
    elements.append(Paragraph(f"<b>User Selected Fault:</b> {complaint.get('fault1', '')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>User Selected Fault:</b> {complaint.get('fault2', '')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>User Selected Fault:</b> {complaint.get('fault3', '')}", styles["Normal"]))


    # CNN Results
    elements.append(Paragraph(f"<b>CNN Detected Issue:</b> {complaint.get('cnn_result', '')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>CNN Confidence:</b> {complaint.get('confidence', '')}%", styles["Normal"]))

    # Employee & Status
    elements.append(Paragraph(f"<b>Assigned Employee:</b> {complaint.get('employee_name', '')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Suggestion:</b> {complaint.get('suggestion', '')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Status:</b> {complaint.get('status', '')}", styles["Normal"]))

    elements.append(Spacer(1, 0.5 * inch))

    # Add Complaint Image (if exists)
    if complaint.get('image_path'):
        image_path = os.path.join(app.root_path, "static", complaint['image_path'])

        if os.path.exists(image_path):
            elements.append(Paragraph("<b>Complaint Image:</b>", styles["Heading2"]))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(RLImage(image_path, width=4 * inch, height=3 * inch))

    # Build PDF
    doc.build(elements)

    # Download file
    return send_file(pdf_path, as_attachment=True)

# =============================
# RUN APP
# =============================
if __name__ == "__main__":
    app.run(debug=True)