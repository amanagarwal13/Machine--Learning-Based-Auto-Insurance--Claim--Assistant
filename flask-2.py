from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, send_file
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from werkzeug.utils import secure_filename
import uuid
import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether, HRFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import numpy as np
from flask import session
import random
from datetime import datetime


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

damage_types = {
    0: "Dent",
    1: "Scratch",
    2: "Crack",
    3: "Glass Shatter",
    4: "Lamp Broken",
    5: "Tire Flat"
}

# Function to add watermark to every page


def generate_report(image_paths,registration_number, owners_name, vin, cubic_capacity, fuel_type, car_name_model, customer_story, policy_number, process_claim):
    # Load car parts model from pickle file
    car_parts = {0: 'Quarter-panel', 1: 'Front-wheel', 2: 'Back-window', 3: 'Trunk', 4: 'Front-door', 5: 'Rocker-panel', 6: 'Grille', 7: 'Windshield', 8: 'Front-window', 9: 'Back-door', 10: 'Headlight', 11: 'Back-wheel', 12: 'Back-windshield', 13: 'Hood', 14: 'Fender', 15: 'Tail-light', 16: 'License-plate', 17: 'Front-bumper', 18: 'Back-bumper', 19: 'Mirror', 20: 'Roof'}

    damage_types = {
        0: "Dent",
        1: "Scratch",
        2: "Crack",
        3: "Glass Shatter",
        4: "Lamp Broken",
        5: "Tire Flat"
    }

    car_part_damage_info = {}
    for car_part_name in car_parts.values():
        car_part_damage_info[car_part_name] = {
                'total_damage_percentage': 0,
                'damage_types': set(),
                'sides_with_damage': set()
            }
        
    with open("OD_cfg_newdataset.pickle", "rb") as f:
        cfg_car_parts = pickle.load(f)

    # Load damage model from pickle file
    with open("OD_cfg.pickle", "rb") as f:
        cfg_damage = pickle.load(f)

    cfg_car_parts.MODEL.DEVICE = "cpu"
    cfg_car_parts.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_car_parts.MODEL.WEIGHTS = "model_final_new_dataset.pth"
    cfg_car_parts.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    car_part_predictor = DefaultPredictor(cfg_car_parts)

    cfg_damage.MODEL.DEVICE = "cpu"
    cfg_damage.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_damage.MODEL.WEIGHTS = "model_final.pth"
    cfg_damage.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    damage_predictor = DefaultPredictor(cfg_damage)

    elements = []
    doc = SimpleDocTemplate("templates/motor_claim_form.pdf", pagesize=letter)

    # Get sample styles
    styles = getSampleStyleSheet()

    # Define logo image
    logo_image = Image('logo.jpg', width=3*inch, height=1*inch)  # Adjust dimensions as needed
    logo_wrap = KeepTogether([logo_image])
    elements.append(logo_wrap)

    # Add the logo text
    logo_text = "www.pixelatedproof.co Toll Free No. 18001035499"
    logo_style = ParagraphStyle('Logo', parent=styles['Heading3'], alignment=1)
    logo = Paragraph(logo_text, logo_style)
    elements.append(logo)
    elements.append(Spacer(1, 12))

    # Add the form title
    title_text = "Motor - Claim Form"
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=1)
    title = Paragraph(title_text, title_style)
    elements.append(title)
    elements.append(Spacer(1, 24))

    # Add a horizontal line
    horizontal_line = HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.black)
    elements.append(horizontal_line)

    # Add the insured details section
    insured_details_text = "Insured Details"
    insured_details_style = ParagraphStyle('InsuredDetails', parent=styles['Heading2'], alignment=1)
    insured_details = Paragraph(insured_details_text, insured_details_style)
    elements.append(insured_details)
    elements.append(Spacer(1, 12))  # Add space after the paragraph


    # Add a table for insured details
    insured_data = [
        ['Report ID','','Report Generated On',''],
        ['Insured Name', '', 'Policy No.', ''],
        ['Registration Number', '', '', ''],
        ['VIN', '', 'Cubic Capacity (in Litres)', ''],
        ['Car Name and Model', ''],
        ['Incident Description','']
    ]

    report_uid = uuid.uuid4()
    report_uid=str(report_uid)
    report_id = random.randint(10000, 99999)
    # Fetch the current date and time
    current_datetime = datetime.now()
    current_datetime_without_microseconds = current_datetime.replace(microsecond=0)

    insured_data[0][3] = current_datetime_without_microseconds
    insured_data[0][1]=report_id
    
    insured_data[1][1] = owners_name
    insured_data[1][3] = policy_number
    insured_data[2][1]=registration_number
    insured_data[3][1]=vin
    insured_data[3][3]=cubic_capacity
    insured_data[4][1]=car_name_model
    insured_data[5][1]=customer_story
    
    
    insured_table = Table(insured_data, colWidths=[2 * inch, 2 * inch, 2 * inch, 2 * inch])
    insured_table.setStyle(TableStyle([
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    elements.append(insured_table)

    elements.append(Spacer(1, 20)) 
    # Add a title
    title_text = "Vehicle Damage Report"
    title_style = ParagraphStyle('Title', fontName="Helvetica-Bold", fontSize=16, alignment=1)
    title = Paragraph(title_text, title_style)
    elements.append(title)
    elements.append(Spacer(1, 25)) 

    # Iterate over detected car parts
    y_position = 10 * inch
    # Iterate over each image
    tire_parts = [1, 11]
    # Iterate over each image
    for side, image_path in zip(["Front", "Back", "Left", "Right"], image_paths):
        # Load and process the image
        image = cv2.imread(image_path)

        # Perform object detection for car parts and damage
        car_part_outputs = car_part_predictor(image)
        damage_outputs = damage_predictor(image)

        # Add side information to the report
        elements.append(Paragraph(f"Side: {side}", styles['Title']))
        elements.append(Spacer(1, 12))

        # Iterate over detected car parts
        # Iterate over detected car parts
        for car_part_idx, car_part_instance in enumerate(car_part_outputs["instances"].pred_masks):
            # Perform necessary operations for each car part
            car_part_mask = car_part_instance.cpu().numpy().astype(bool)
            car_part_idx = car_part_outputs["instances"].pred_classes[car_part_idx].item()
            car_part_name = car_parts[car_part_idx]

            # Find overlapping damage regions
            damage_masks = []
            damage_names = []
            for damage_idx, damage_instance in enumerate(damage_outputs["instances"].pred_masks):
                damage_mask = damage_instance.cpu().numpy().astype(bool)
                if calc_overlap_area(car_part_mask, damage_mask) > 0:
                    damage_masks.append(damage_mask)
                    damage_idx = damage_outputs["instances"].pred_classes[damage_idx].item()
                    damage_name = damage_types[damage_idx]
                    
                    # For Front-wheel and Back-wheel, only consider "Tire Flat" damage
                    if car_part_idx in tire_parts:
                        if damage_name == "Tire Flat":
                            damage_names.append(damage_name)
                        # Filter out "Tire Flat" damage for non-tire car parts
                    elif damage_name == "Tire Flat" and car_part_idx not in tire_parts:
                        continue
                    
                    damage_names.append(damage_name)
            
            # Calculate the percentage of damage for the car part
            damage_percentage = calc_damage_percentage(car_part_mask, damage_masks)
            car_part_name = car_parts[car_part_idx]
            car_part_damage_info[car_part_name]['total_damage_percentage'] += damage_percentage
            car_part_damage_info[car_part_name]['damage_types'].update(damage_names)
            car_part_damage_info[car_part_name]['sides_with_damage'].add(side)

            if damage_names:
                # Add car part details to elements
                car_part_style = ParagraphStyle('CarPart', fontName="Helvetica-Bold", fontSize=12)
                damage_style = ParagraphStyle('Damage', fontName="Helvetica", fontSize=10)
        
                elements.append(Paragraph(f"Car Part: {car_part_name}", car_part_style))
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(f"Detected Damage: {', '.join(damage_names)}", damage_style))
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(f"Damage Percentage: {damage_percentage:.2f}%", damage_style))
                elements.append(Spacer(1, 12))  # Add extra spacing between car parts


    # Add a title for the comprehensive car part damage analysis
    title_text = "Comprehensive Car Part Damage Analysis"
    title_style = ParagraphStyle('Title', fontName="Helvetica-Bold", fontSize=16, alignment=1)
    title = Paragraph(title_text, title_style)
    elements.append(title)
    elements.append(Spacer(1, 25))
    
    for car_part_name, damage_info in car_part_damage_info.items():
        total_damage_percentage = damage_info['total_damage_percentage']
        damage_types = ', '.join(damage_info['damage_types'])
        sides_with_damage = ', '.join(damage_info['sides_with_damage'])

        max_possible_damage_percentage = min(total_damage_percentage, 100)
        # Determine if replacement is needed based on total damage percentage
        if max_possible_damage_percentage > 50:
            replacement_required = "Yes"
        else:
            replacement_required = "No"
        repair_required_list = []
        for damage_type in damage_info['damage_types']:
            if damage_type == "Dent":
                repair_required_list.append("Denting")
            elif damage_type == "Scratch":
                repair_required_list.append("Painting")
            elif damage_type in ["Crack", "Glass Shatter", "Lamp Broken"]:
                repair_required_list.append("Replacement")
            elif damage_type == "Tire Flat":
                repair_required_list.append("Puncture Repair")
            else:
                repair_required_list.append("Unknown")
            
        
        if max_possible_damage_percentage > 50:
            repair_required_list = [ " No repair can be performed"]
        
        # Combine repair types into a single string
        repair_required_str = ', '.join(repair_required_list)
        # Add the comprehensive damage report for the car part
        car_part_style = ParagraphStyle('CarPart', fontName="Helvetica-Bold", fontSize=12)
        analysis_style = ParagraphStyle('Analysis', fontName="Helvetica", fontSize=10)
        
        elements.append(Paragraph(f"Car Part: {car_part_name}", car_part_style))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"Total Damage Percentage: {max_possible_damage_percentage:.2f}%", analysis_style))
        elements.append(Paragraph(f"Detected Damage Types: {damage_types}", analysis_style))
        elements.append(Paragraph(f"Sides containing the car part: {sides_with_damage}", analysis_style))
        elements.append(Paragraph(f"Replacement Required: {replacement_required}", analysis_style))

        elements.append(Paragraph(f"Repair Required: {repair_required_str}", analysis_style))

        elements.append(Spacer(1, 12))  # Add extra spacing between car parts
    
    print("Report generated: car_damage_report.pdf")
    
    def add_watermark(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 60)
        canvas.setFillGray(0.9)  # Adjust the transparency of the watermark
        
        # Draw the watermark at specific coordinates (e.g., (400, 400))
        #canvas.rotate(45)
        canvas.drawString(100, 300, "Pixelated Proof")  # Draw the watermark at specified coordinates
        

        # Add footer with darkened text
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(50, 30, "Pixelated Proof")  # Adjust coordinates and text as needed
        canvas.drawString(50, 20, report_uid)  # Adjust coordinates and text as needed
        
        canvas.restoreState()
        
    doc.build(elements, onFirstPage=add_watermark, onLaterPages=add_watermark)

    
def on_image(image_path, predictor, save_path):
    img = cv2.imread(image_path)
    outputs = predictor(img)
    
    pred_classes = outputs["instances"].get("pred_classes").cpu().numpy()
    for pred_class in pred_classes:
        damage_type = damage_types.get(pred_class, "Unknown")
        print("Predicted Damage Type:", damage_type)
    
    v = Visualizer(img[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save the image with predicted segments
    output_img = v.get_image()[:, :, ::-1]  # Convert BGR to RGB
    cv2.imwrite(save_path, output_img)

def process(side, file_path):
    model_weights_path = "model_final.pth"
    
    cfg_save_path = "OD_cfg.pickle"
    with open(cfg_save_path, 'rb') as f:
        cfg = pickle.load(f)

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = model_weights_path  # path to the model we just trained

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    save_path = f"D:/FirebaseJSON/static/{side}_output.jpg"
    on_image(file_path, predictor, save_path)


def detect_damage(image, predictor):
    # Perform damage detection on the image
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.get("pred_classes").numpy()
    damage_detected = [damage_types.get(pred_class, "Unknown") for pred_class in pred_classes]
    return instances, damage_detected

def visualize_damage_detections(image, instances_damage):
    v = Visualizer(image[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(instances_damage.to("cpu"))
    combined_image = out.get_image()[:, :, ::-1]
    return combined_image

def calc_overlap_area(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection)

def calc_damage_percentage(car_part_mask, damage_masks):
    car_part_area = np.sum(car_part_mask)
    total_overlap_area = 0
    for damage_mask in damage_masks:
        overlap_area = calc_overlap_area(car_part_mask, damage_mask)
        total_overlap_area += overlap_area
    damage_percentage = (total_overlap_area / car_part_area) * 100
    return damage_percentage

car_parts = {0: 'Quarter-panel', 1: 'Front-wheel', 2: 'Back-window', 3: 'Trunk', 4: 'Front-door', 5: 'Rocker-panel', 6: 'Grille', 7: 'Windshield', 8: 'Front-window', 9: 'Back-door', 10: 'Headlight', 11: 'Back-wheel', 12: 'Back-windshield', 13: 'Hood', 14: 'Fender', 15: 'Tail-light', 16: 'License-plate', 17: 'Front-bumper', 18: 'Back-bumper', 19: 'Mirror', 20: 'Roof'}

damage_types = {
    0: "Dent",
    1: "Scratch",
    2: "Crack",
    3: "Glass Shatter",
    4: "Lamp Broken",
    5: "Tire Flat"
}



# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate('C:/Users/MSI1/Desktop/major project 2 final/FirebaseJSON/insuranceclaimdb-firebase-adminsdk-nqj5n-dbf0c37136.json')
    firebase_admin.initialize_app(cred, {'storageBucket': 'insuranceclaimdb.appspot.com'  })
bucket_name = 'insuranceclaimdb.appspot.com'
db = firestore.client()
bucket = storage.bucket(app=firebase_admin.get_app(), name=bucket_name)

class User(UserMixin):
    def __init__(self, user_id=None, username=None, password=None):
        if user_id is None:
            user_id = str(uuid.uuid4())  # Generate a unique user ID if not provided
        self.id = user_id
        self.username = username
        self.password = password

    def save_to_db(self):
        user_ref = db.collection('users').document(self.id)
        user_ref.set({
            'id': self.id,
            'username': self.username,
            'password': self.password,
        })

    def get_id(self):
        return str(self.id)  # Return the ID as a string

    @staticmethod
    def find_by_id(user_id):
        user_doc = db.collection('users').document(user_id).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            return User(user_id=user_data['id'], username=user_data['username'], password=user_data['password'])
        return None
    
    @staticmethod
    def find_by_username(username_or_id):
        users_ref = db.collection('users').where('username', '==', username_or_id).stream()
        for user_doc in users_ref:
            user_data = user_doc.to_dict()
            return User(user_id=user_data['id'], username=user_data['username'], password=user_data['password'])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.find_by_id(user_id)

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        user = User(username=username, password=hashed_password)
        user.save_to_db()

        flash('Your account has been created!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.find_by_username(username)
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            print(f"User {username} logged in successfully")
            return render_template('upload.html')    # Redirect to upload page on successful login
        else:
            flash('Login failed. Please check your username and password.', 'danger')

    return render_template('login.html')

@app.route('/uploadfile', methods=['GET', 'POST'])
@login_required
def uploadfile():
    if request.method == 'POST':
        
        registration_number = request.form['registrationNumber']
        owners_name = request.form['ownersName']
        vin = request.form['VIN']
        cubic_capacity = request.form['cubicCapacity']
        fuel_type = request.form['fuelType']
        car_name_model = request.form['carNameModel']
        customer_story = request.form['customerStory']
        policy_number = request.form['policyNumber']
        process_claim = request.form.get('processClaim', False)
        
        file_paths = {}  # Dictionary to store file paths for different sides
        file_paths2=[]
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        # Iterate over the four sides and save the files
        for side in ['front', 'back', 'left', 'right']:
            file = request.files.get(f'{side}File')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = f"D:/FirebaseJSON/static/{side}_input.jpg"
                file.save(file_path)
                file_paths[side] = file_path
                file_paths2.append(file_path)
            else:
                flash(f'Invalid file type for {side} side. Allowed extensions are: png, jpg, jpeg')
                return redirect(request.url)

        if file_paths:  # Proceed only if files were uploaded for all sides
            process_claim = request.form.get('processClaim')  # Retrieve the checkbox value
            if process_claim:
                user = User.find_by_id(current_user.get_id())
                user_doc_ref = db.collection('users').document(user.id)
                user_data = user_doc_ref.get()

                if not user_data.exists:
                    user_doc_ref.set({'claimcounter': 0})
                    print("Created 'claimcounter' field in the user document.")

                current_count = user_data.to_dict().get('claimcounter', 0)
                user_doc_ref.update({'claimcounter': current_count + 1})
                print("'claimcounter' incremented.")
            
            # Store filenames in session
            session['uploaded_filenames'] = file_paths
            print(file_paths2)
            # Process each side separately
            for side, file_path in file_paths.items():
                process(side, file_path)
            
            generate_report(file_paths2,registration_number, owners_name, vin, cubic_capacity, fuel_type, car_name_model, customer_story, policy_number, process_claim)
            
            return redirect(url_for('display_uploaded_file', filename=filename))
        
        return redirect(request.url)  # Redirect if files were not uploaded
    
    return render_template('upload.html')


    

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/uploads/<filename>')
@login_required
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download_report', methods=['GET'])
def download_report():
    # Path to your PDF file
    pdf_path = 'D:/FirebaseJSON/templates/motor_claim_form.pdf'
    return send_file(pdf_path, as_attachment=True)

@app.route('/display/<filename>')
@login_required
def display_uploaded_file(filename):
    return render_template('display.html', filename=filename)

@app.route('/contact')
def contact():
    return render_template('contact.html')\

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.run(debug=True, use_reloader=False)
