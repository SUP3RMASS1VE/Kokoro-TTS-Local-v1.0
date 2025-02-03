# A Pinokio script for Kokoro-TTS
----------------------------------

# Kokoro TTS Local

**Kokoro TTS Local** is a cutting-edge text-to-speech (TTS) synthesis program that utilises deep learning to generate high-quality speech from text. This tool provides multiple voice options, speed adjustments, and support for various languages. The best part? It runs locally, ensuring fast, offline speech generation directly from your computer.

---

## Key Features

- **Voice Selection**: Choose from a diverse range of voices, including various accents and languages like English and more.
- **Speech Speed Control**: Adjust the speech speed anywhere from **0.5x** to **4x** to suit your preference.
- **Phoneme Output**: View the phoneme sequence for each generated speech.
- **GPU/CPU Support**: Automatically switches to CPU if GPU support isn’t available, ensuring smooth operation.
- **Local Operation**: Runs entirely on your local machine without needing an internet connection.
- **High-Quality Audio**: Generates **24kHz** high-quality audio for clear and natural speech.

---

## Requirements

- Python 3.x
- PyTorch (with CUDA support, if available)
- Gradio
- Kokoro
- tqdm
- scipy

### Installation

You can easily install the required dependencies using `pip`. If you plan to use GPU support, make sure to install the correct **CUDA** version.

---

## Usage Guide

### Step 1: Set Up the Environment

1. Install all necessary dependencies as listed.
2. Ensure your machine has GPU access for optimal performance. If not, the system will default to the CPU.
3. Get the necessary files and configurations for the Kokoro models and pipelines.

### Step 2: Running the TTS Application

Once everything is set up, run the following command to launch the application:

```bash
python app.py
```

This will open a web interface powered by **Gradio**, where you can:

- **Enter Text**: Input the text you’d like to convert to speech.
- **Select Voice**: Pick from a variety of voices, each with distinct accents and styles.
- **Adjust Speed**: Use the slider to control the speech speed.
- **Generate Speech**: Click the “Generate Speech” button to create your audio.
- **Listen to Audio**: The generated speech will play back, and the phoneme sequence will appear.

### Step 3: Output

The generated speech will be saved as a **.wav** file in the `outputs/` folder with a timestamp. The phoneme sequence for your input text will also be displayed below the audio.

---

## Voice Choices

Here are the available voices to choose from:

**English (US)**  
- 🇺🇸 💺 **Heart** ❤️  
- 🇺🇸 💺 **Bella** 🔥  
- 🇺🇸 💺 **Nicole** 🎧  
- 🇺🇸 💺 **Aoede**  
- 🇺🇸 💺 **Kore**  
- 🇺🇸 💺 **Sarah**  
- 🇺🇸 💺 **Nova**  
- 🇺🇸 💺 **Sky**  
- 🇺🇸 💺 **Alloy**  
- 🇺🇸 💺 **Jessica**  
- 🇺🇸 💺 **River**  
- 🇺🇸 👨 **Michael**  
- 🇺🇸 👨 **Fenrir**  
- 🇺🇸 👨 **Puck**  
- 🇺🇸 👨 **Echo**  
- 🇺🇸 👨 **Eric**  
- 🇺🇸 👨 **Liam**  
- 🇺🇸 👨 **Onyx**  
- 🇺🇸 👨 **Santa**  
- 🇺🇸 👨 **Adam**  

**English (UK)**  
- 🇬🇧 💺 **Emma**  
- 🇬🇧 💺 **Isabella**  
- 🇬🇧 💺 **Alice**  
- 🇬🇧 💺 **Lily**  
- 🇬🇧 👨 **George**  
- 🇬🇧 👨 **Fable**  
- 🇬🇧 👨 **Lewis**  
- 🇬🇧 👨 **Daniel**  

Each voice offers a distinct style and accent, providing you with plenty of options for your speech generation needs.

---

## Code Explanation

1. **Model Loading**: The TTS model is loaded onto your system’s GPU or CPU, depending on available resources. It uses the Kokoro TTS system for advanced speech synthesis.
2. **Pipeline Setup**: Separate pipelines handle different languages and models, offering flexibility in voice and language choices.
3. **Text Preprocessing**: Input text is processed in manageable chunks to fit within the character limits of the TTS system.
4. **Audio Generation**: Once the text is processed, speech is generated and saved as an audio file.
5. **Web Interface**: The Gradio interface allows users to interact with the system by entering text, selecting voices, adjusting speed, and hearing the generated speech.

---

## Troubleshooting

- **GPU Not Detected**: Ensure that the correct CUDA drivers are installed and your GPU is compatible with PyTorch.
- **Errors During Speech Generation**: If this occurs, it might be due to overly long text or special characters. Try simplifying the input or shortening the text.

---


