import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import time

# Record audio function
def record_audio(word, duration=2, fs=44100):
    print(f"Recording {word} for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    np.save(f"{word}.npy", audio)
    print(f"Recording {word} complete.")

# Record audio button callback
def on_record_click(word_entry):
    word = word_entry.get()
    if not word:
        print("Please enter a word before recording.")
        return
    record_audio(word)

# Train model function (to be implemented)
def train_model():
    print("Training model...")
    # Implement your model training here
    print("Model training complete.")

# Main GUI function
def main():
    root = tk.Tk()
    root.title("Audio Recorder")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    word1_label = ttk.Label(frame, text="Word 1:")
    word1_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    word1_entry = ttk.Entry(frame)
    word1_entry.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

    record1_button = ttk.Button(frame, text="Record", command=lambda: on_record_click(word1_entry))
    record1_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

    word2_label = ttk.Label(frame, text="Word 2:")
    word2_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

    word2_entry = ttk.Entry(frame)
    word2_entry.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

    record2_button = ttk.Button(frame, text="Record", command=lambda: on_record_click(word2_entry))
    record2_button.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)

    train_button = ttk.Button(frame, text="Train Model", command=train_model)
    train_button.grid(row=2, column=0, columnspan=3, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()

