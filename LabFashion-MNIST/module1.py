import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("fashion_mnist_model.h5")

# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img_arr = np.array(img).astype("float32") / 255
    img_arr = np.expand_dims(img_arr, axis=(0, -1))
    prediction = model.predict(img_arr)
    top_3_indices = np.argsort(prediction[0])[-3:][::-1]
    top_3_confidences = prediction[0][top_3_indices] * 100
    top_3_classes = [class_names[i] for i in top_3_indices]
    return top_3_classes, top_3_confidences

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        
        label_image.config(image=img)
        label_image.image = img
        
        top_3_classes, top_3_confidences = predict_image(file_path)
        result_text = "Top 3 Predictions:\n"
        for i in range(3):
            result_text += f"{i+1}. {top_3_classes[i]} - {top_3_confidences[i]:.2f}%\n"
        label_result.config(text=result_text)


# Create the main window
root = tk.Tk()
root.title("Fashion-MNIST Classifier")

# Add the main frame
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Add the "Open" button
button_open = tk.Button(frame, text="Open Image", command=open_image)
button_open.pack(pady=(0, 10))

# Add the image label
label_image = tk.Label(frame)
label_image.pack(pady=(0, 10))

# Add the result label
label_result = tk.Label(frame, text="")
label_result.pack()

# Start the main loop
root.mainloop()

