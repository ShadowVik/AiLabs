import os
import tkinter as tk
import pyaudio
import wave
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('spoken_digit_recognition.h5')

# Record audio function
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    predict_digit(WAVE_OUTPUT_FILENAME)

# Predict digit function
def predict_digit(file_path):
    # Load and preprocess the recorded audio
    audio_data, sample_rate = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
    audio_data = tf.squeeze(audio_data)
    audio_data = tf.pad(audio_data, paddings=((0, 30000 - tf.shape(audio_data)[0]),))
    audio_data = tf.expand_dims(audio_data, axis=-1)
    audio_data = tf.expand_dims(audio_data, axis=0)

    # Predict the digit using the loaded model
    prediction = model.predict(audio_data)
    digit = np.argmax(prediction)

    print("Predicted digit:", digit)

# Create the GUI
root = tk.Tk()
root.title("Spoken Digit Recognition")
root.geometry("300x200")

frame = tk.Frame(root)
frame.pack(pady=20)

button = tk.Button(frame, text="Press and Speak", command=record_audio)
button.pack()

root.mainloop()

