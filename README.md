Kokoro TTS Local
Kokoro is a high-quality text-to-speech (TTS) synthesis program that leverages deep learning techniques. This project allows users to generate high-fidelity speech from text input, with several voice options and speed adjustments. The model supports multiple languages, and the system can run locally, making it fast and easy to generate speech right from your computer.

Features
Voice Selection: Choose from a variety of voices, including options from different languages such as English and others.
Speech Speed Control: Adjust the speed of the generated speech from 0.5x to 4x to match your needs.
Phoneme Output: Get the phoneme sequence for the generated speech.
GPU/CPU Support: Automatically falls back to CPU if GPU support is unavailable.
Local Operation: Runs entirely locally on your machine without requiring an internet connection.
High-Quality Audio: Produces high-quality speech audio at 24kHz.
Requirements
Python 3.x
PyTorch (with CUDA support if available)
Gradio
kokoro
tqdm
scipy
You can install the necessary dependencies via pip:
Ensure you have the correct CUDA version installed if you want GPU support.

Usage
Step 1: Set up the environment
Install the required dependencies as mentioned above.
Make sure your machine has access to a GPU for optimal performance. If not, the model will default to using the CPU.
Ensure that you have the necessary files and configurations for Kokoro’s models and pipelines.
Step 2: Running the TTS Application
Once the environment is set up, you can run the script to launch the application.

python app.py
The application will launch a web interface powered by Gradio. You can:

Enter Text: Input the text you want to convert into speech.
Select Voice: Choose a voice from the dropdown menu. There are several voices available, with different accents and styles.
Adjust Speed: Use the slider to control the speed of the speech generation.
Generate Speech: Click the "Generate Speech" button to create the audio output.
Listen to Audio: The generated speech will play back to you, and the phoneme sequence will be shown below.
Step 3: Output
The generated speech will be saved as a .wav file in the outputs/ folder, with a timestamp in the filename. Additionally, the phoneme sequence corresponding to the input text will be displayed.

Voice Choices
The following voices are available for selection:

🇺🇸 🚺 Heart ❤️
🇺🇸 🚺 Bella 🔥
🇺🇸 🚺 Nicole 🎧
🇺🇸 🚺 Aoede
🇺🇸 🚺 Kore
🇺🇸 🚺 Sarah
🇺🇸 🚺 Nova
🇺🇸 🚺 Sky
🇺🇸 🚺 Alloy
🇺🇸 🚺 Jessica
🇺🇸 🚺 River
🇺🇸 🚹 Michael
🇺🇸 🚹 Fenrir
🇺🇸 🚹 Puck
🇺🇸 🚹 Echo
🇺🇸 🚹 Eric
🇺🇸 🚹 Liam
🇺🇸 🚹 Onyx
🇺🇸 🚹 Santa
🇺🇸 🚹 Adam
🇬🇧 🚺 Emma
🇬🇧 🚺 Isabella
🇬🇧 🚺 Alice
🇬🇧 🚺 Lily
🇬🇧 🚹 George
🇬🇧 🚹 Fable
🇬🇧 🚹 Lewis
🇬🇧 🚹 Daniel
Each voice has a unique style and accent, providing a wide range of options for your speech generation.

Code Explanation
Model Loading: The model is loaded onto either GPU or CPU depending on your system’s capabilities. It uses the Kokoro TTS system for deep learning-based speech synthesis.
Pipeline Setup: Different languages and models are handled by separate pipelines, allowing flexibility in voice and language selection.
Text Preprocessing: The input text is split into manageable chunks to fit the character limit.
Audio Generation: After processing the text, the TTS system generates speech, which is saved as an audio file.
Web Interface: Gradio is used to create an interactive UI where users can enter text, select voices, adjust speech speed, and listen to the generated audio.
Troubleshooting
If you experience any issues with GPU processing, the application will automatically fall back to CPU processing.

GPU not detected: Ensure you have the correct CUDA drivers installed and your GPU is supported by PyTorch.
Errors during speech generation: This might occur if the text is too long or contains special characters. Try reducing the length or simplifying the text.
