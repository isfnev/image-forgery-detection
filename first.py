import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import ImageChops, ImageEnhance
from tensorflow.keras.models import load_model

def process_image():
    file_path = filedialog.askopenfilename()
    print(file_path)
    if file_path:
        image = Image.open(file_path)
        
        img = ImageTk.PhotoImage(image)
        img_label.config(image=img)
        img_label.image = img
        
        result_label.config(text='')
        process_button.config(state=tk.NORMAL, command=lambda: display_result(file_path))

def display_result(path):
    image = prepare_image(path)
    image = image.reshape(-1,128,128,3)
    
    result_label.config(text="Image processed successfully!", fg="green")
    y_pred = model.predict(image)

    if np.argmax(y_pred,axis=1)[0]:
        result_label.config(text="It's a Real Image", fg='green')
    else:
        result_label.config(text="It's a Fake Image", fg='red')
    process_button.config(state=tk.DISABLED)
    
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image


def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

root = tk.Tk()
root.title("Image Forgery Detection")

#root.geometry("300x100")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=20)

image_size = (128, 128)
model= load_model('model_casia_run1.keras')

upload_button = tk.Button(frame, text="Upload Image", command=process_image, bg="#007BFF", fg="white", font=("Helvetica", 12, "bold"))
upload_button.pack(padx=100,pady=20)

img_label = tk.Label(frame, bg="#f0f0f0")
img_label.pack(pady=10)

process_button = tk.Button(root, text="Process Image", bg="#28a745", fg="white", font=("Helvetica", 12, "bold"), state=tk.DISABLED)
process_button.pack(pady=10)

result_label = tk.Label(root, text="", bg="#f0f0f0", font=("Helvetica", 12))
result_label.pack(pady=10)

root.mainloop()

