import cv2
import numpy as np
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import pickle
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Toplevel
from tkinter import Canvas

# Initialize the FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known embeddings from a file or create a new dictionary
try:
    with open("embedding_database.pkl", "rb") as f:
        embedding_database = pickle.load(f)
    if not isinstance(embedding_database, dict):
        embedding_database = {}
except FileNotFoundError:
    embedding_database = {}
    # Create a new database file if it doesn't exist
    with open("embedding_database.pkl", "wb") as f:
        pickle.dump(embedding_database, f)

# Function to recognize the face
def recognize_face(new_embedding, embedding_database, threshold=1.0):
    if len(embedding_database) == 0:
        return "Unknown"

    embeddings = np.array(list(embedding_database.values()))
    names = list(embedding_database.keys())
    distances = np.linalg.norm(embeddings - new_embedding, axis=1)
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]

    if min_distance < threshold:
        identity = names[min_distance_idx]
    else:
        identity = "Unknown"

    return identity

# Function to display image with detected or recognized names
def display_image_with_name(img, window_title="Image"):
    # Convert image from BGR to RGB for displaying
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Create a new window to display the image
    top = Toplevel()
    top.title(window_title)
    top.configure(bg="#2C3E50")
    top.geometry("600x600")

    img_label = tk.Label(top, bg="#2C3E50")
    img_label.pack(pady=10)

    img_display = pil_img.copy()
    img_display.thumbnail((min(top.winfo_screenwidth() // 2, 500), min(top.winfo_screenheight() // 2, 500)))
    img_tk = ImageTk.PhotoImage(img_display)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

# Function to add a new face to the embedding database
def add_new_face(image_path, name):
    # Load the image from the given path
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", f"Unable to load image at {image_path}")
        return

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(img)

    if isinstance(faces, dict):
        face_added = False
        for key in faces:
            face = faces[key]
            facial_area = face["facial_area"]

            # Crop the face using the detected bounding box
            x1, y1, x2, y2 = facial_area
            face_crop = img[y1:y2, x1:x2]

            # Resize the cropped face to a standard size (e.g., 160x160 pixels)
            face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)

            # Convert to PIL Image for compatibility with FaceNet
            face_img = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

            # Preprocess face image for FaceNet
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            face_tensor = preprocess(face_img).unsqueeze(0).to(device)

            # Get the embedding for the face
            with torch.no_grad():
                embedding = model(face_tensor).cpu().numpy().flatten()

            # Add the new face embedding to the database
            embedding_database[name] = embedding
            face_added = True

            # Draw the bounding box and name on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if face_added:
            with open("embedding_database_temp.pkl", "wb") as f:
                pickle.dump(embedding_database, f)
            os.replace("embedding_database_temp.pkl", "embedding_database.pkl")
            messagebox.showinfo("Success", f"Face(s) enrolled as '{name}'")

            # Display the image with name(s)
            display_image_with_name(img, f"Enrolled: {name}")
        else:
            messagebox.showerror("Error", "No face detected in the provided image.")
    else:
        messagebox.showerror("Error", "No face detected in the provided image.")

# Function to add a new face using the webcam
def add_new_face_from_camera(name):
    # Use the webcam to capture an image
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera")
        return
    ret, img = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Unable to capture image from camera")
        return

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(img)

    if isinstance(faces, dict):
        face_added = False
        for key in faces:
            face = faces[key]
            facial_area = face["facial_area"]

            # Crop the face using the detected bounding box
            x1, y1, x2, y2 = facial_area
            face_crop = img[y1:y2, x1:x2]

            # Resize the cropped face to a standard size (e.g., 160x160 pixels)
            face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)

            # Convert to PIL Image for compatibility with FaceNet
            face_img = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

            # Preprocess face image for FaceNet
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            face_tensor = preprocess(face_img).unsqueeze(0).to(device)

            # Get the embedding for the face
            with torch.no_grad():
                embedding = model(face_tensor).cpu().numpy().flatten()

            # Add the new face embedding to the database
            embedding_database[name] = embedding
            face_added = True

            # Draw the bounding box and name on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if face_added:
            with open("embedding_database.pkl", "wb") as f:
                pickle.dump(embedding_database, f)
            messagebox.showinfo("Success", f"Face(s) enrolled as '{name}'")

            # Display the image with name(s)
            display_image_with_name(img, f"Enrolled: {name}")
        else:
            messagebox.showerror("Error", "No face detected in the captured image.")
    else:
        messagebox.showerror("Error", "No face detected in the captured image.")

# Function to open a file dialog and select an image
def select_image():
    file_path = filedialog.askopenfilename()
    return file_path

# Function to handle adding a new face
def handle_add_face():
    name_window = Toplevel()
    name_window.title("Enter Name")
    name_window.geometry("300x150")
    name_window.configure(bg="#2A3E4C")

    tk.Label(name_window, text="Enter Name:", bg="#2A3E4C", fg="#ECF0F1", font=("Helvetica", 12)).pack(pady=10)
    name_entry = tk.Entry(name_window, font=("Helvetica", 12))
    name_entry.pack(pady=5)

    def handle_name():
        name = name_entry.get().strip()
        if name:
            response = messagebox.askquestion("Add Face", "Do you want to add a face from an image file? Click 'No' to use the camera.")
            if response == 'yes':
                image_path = select_image()
                if image_path:
                    add_new_face(image_path, name)
            else:
                add_new_face_from_camera(name)
        else:
            messagebox.showerror("Error", "Please enter a name for the person.")
        name_window.destroy()

    tk.Button(name_window, text="Submit", command=handle_name, bg="#3498DB", fg="white", font=("Helvetica", 12)).pack(pady=10)

# Function to detect and recognize faces
def handle_recognize_face():
    response = messagebox.askquestion("Recognition Option", "Do you want to recognize a face from an image file? Click 'No' to use the camera.")
    if response == 'yes':
        image_path = select_image()
        if image_path:
            detect_and_recognize_face(image_path)
    else:
        detect_and_recognize_face()

# Function to detect and recognize faces
def detect_and_recognize_face(image_path=None):
    if image_path:
        # Load the image from the given path
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", f"Unable to load image at {image_path}")
            return
    else:
        # Use the webcam to capture an image
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera")
            return
        ret, img = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Unable to capture image from camera")
            return

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(img)

    if isinstance(faces, dict):
        recognition_results = []
        for key in faces:
            face = faces[key]
            facial_area = face["facial_area"]

            # Crop the face using the detected bounding box
            x1, y1, x2, y2 = facial_area
            face_crop = img[y1:y2, x1:x2]

            # Resize the cropped face to a standard size (e.g., 160x160 pixels)
            face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)

            # Convert to PIL Image for compatibility with FaceNet
            face_img = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

            # Preprocess face image for FaceNet
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            face_tensor = preprocess(face_img).unsqueeze(0).to(device)

            # Get the embedding for the face
            with torch.no_grad():
                embedding = model(face_tensor).cpu().numpy().flatten()

            # Recognize the face
            name = recognize_face(embedding, embedding_database)

            recognition_results.append(name)

            # Draw the bounding box and name on the image
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the image with names
        display_image_with_name(img, "Recognition Results")

        # Show recognition results
        result_message = "\n".join([f"Face {i+1}: {name}" for i, name in enumerate(recognition_results)])
        messagebox.showinfo("Recognition Results", result_message)
    else:
        messagebox.showerror("Error", "No face detected in the image.")

# Create the GUI application
app = tk.Tk()
app.title("Face Recognition System")
app.geometry("600x400")
app.configure(bg="#2A3E4C")

# Header Canvas for Modern Look
header = Canvas(app, height=80, width=600, bg="#1A252F", highlightthickness=0)
header.create_text(300, 40, text="Face Recognition System", fill="#FFFFFF", font=("Helvetica", 20, "bold"))
header.pack()

# UI Elements
frame = tk.Frame(app, bg="#34495E")
frame.pack(pady=20, padx=20, fill="both", expand=True)

add_face_button = tk.Button(frame, text="Add New Face", command=handle_add_face, bg="#3498DB", fg="white", font=("Helvetica", 14, "bold"), activebackground="#2980B9", relief="raised", bd=5)
add_face_button.grid(row=0, column=0, pady=10, padx=10, sticky="w")

recognize_face_button = tk.Button(frame, text="Recognize Face", command=handle_recognize_face, bg="#E74C3C", fg="white", font=("Helvetica", 14, "bold"), activebackground="#C0392B", relief="raised", bd=5)
recognize_face_button.grid(row=1, column=0, pady=10, padx=10, sticky="w")

# Run the application
app.mainloop()
