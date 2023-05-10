import tensorflow as tf
import tensorflow_datasets as tfds

# Load the EMNIST letters dataset
emnist_data = tfds.load('emnist/letters', split='test', as_supervised=True)
emnist_classes = "abcdefghijklmnopqrstuvwxyz"

# Load the EMNIST model weights
emnist_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(27, activation='softmax')
])
emnist_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
emnist_model.load_weights("emnist_letters_model_weights.h5")

# Define a function to recognize handwritten characters
def recognize_character(image):
    # Convert the image to grayscale and resize it to 28x28 pixels
    image = image.convert('L').resize((28, 28))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Invert the colors (black on white)
    image_array = 255 - image_array
    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0
    # Reshape the array to have a single channel (EMNIST images are grayscale)
    image_array = np.reshape(image_array, (1, 28, 28, 1))
    # Use the pre-trained model to predict the character
    prediction = model.predict(image_array)
    # Get the index of the highest probability (i.e., the predicted character)
    character = chr(np.argmax(prediction) + 96)  # Convert index to ASCII character code (a=97)
    # Return the predicted character
    return character

# Define a function to save the current drawing as an image
def save_image(canvas):
    # Get the size of the canvas
    x, y = canvas.winfo_width(), canvas.winfo_height()
    # Create a new image with the same size as the canvas
    image = PIL.Image.new('RGB', (x, y), 'white')
    # Draw the canvas onto the image
    image_draw = PIL.ImageDraw.Draw(image)
    image_draw.bitmap((0, 0), canvas.postscript(colormode='color'), fill='black')
    # Save the image to a file
    filename = 'drawing.png'
    image.save(filename)
    print(f"Saved image as {filename}")

# Define a function to recognize the current drawing on the canvas
def recognize_drawing(canvas):
    # Convert the canvas to an image
    image = PIL.Image.frombytes('RGB', (canvas.winfo_width(), canvas.winfo_height()), canvas.postscript(colormode='color'))
    # Call the recognize_character function to recognize the image
    character = recognize_character(image)
    # Display the recognized character in a message box
    tk.messagebox.showinfo("Recognized character", f"The recognized character is '{character}'")

# Define a function to clear the canvas
def clear_canvas(canvas):
    canvas.delete("all")

# Create the main window
root = tk.Tk()
root.title("Handwriting recognition")

# Create a canvas for drawing
canvas = tk.Canvas(root, bg="white", width=280, height=280)
canvas.pack()

# Create buttons for saving, recognizing, and clearing the drawing
save_button = tk.Button(root, text="Save", command=lambda: save_image(canvas))
recognize_button = tk.Button(root, text="Recognize", command=lambda: recognize_drawing(canvas))
clear_button = tk.Button(root, text="Clear", command=lambda: clear_canvas(canvas))
save_button.pack(side=tk.LEFT, padx=10, pady=10)
recognize_button.pack(side=tk.LEFT, padx=10, pady=10)
clear_button.pack(side=tk.LEFT, padx=10, pady=10)

# Create a function to handle mouse events on the canvas
def handle_mouse_event(event):
    if event.type == tk.EventType.ButtonPress:
    # Save the starting position of the mouse
        canvas.last_x, canvas.last_y = event.x, event.y
    elif event.type == tk.EventType.B1_Motion:
# Draw a line from the previous position to the current position
        canvas.create_line(canvas.last_x, canvas.last_y, event.x, event.y, width=5, fill='black')
# Update the last position to the current position
        canvas.last_x, canvas.last_y = event.x, event.y

canvas.bind("<ButtonPress-1>", handle_mouse_event)
canvas.bind("<B1-Motion>", handle_mouse_event)
root.mainloop()