# Kokoro TTS Local

Welcome to Kokoro, a high-quality text-to-speech synthesis program powered by deep learning. This tool converts any text into high-fidelity speech in just a few seconds. Simply input text, select a voice, adjust the speed, and enjoy the generated audio.
![Screenshot 2025-03-16 190725](https://github.com/user-attachments/assets/dbb36d4b-62d4-4c15-b5b3-bf4c47539e58)
![Screenshot 2025-03-16 190814](https://github.com/user-attachments/assets/1a253b90-6d4b-45fa-aa53-c154b971d7e6)
![Screenshot 2025-03-16 190747](https://github.com/user-attachments/assets/2afc92f4-55a2-4916-861a-bdfb23237416)

## Installation  
### Option 1: Install via [Pinokio](https://pinokio.co).

### Option 1a: Install via [Dione](https://getdione.app)

## Features

- **High-Quality TTS** – Uses the Kokoro model to produce realistic and natural-sounding speech.
- **Multiple Voices** – Choose from a wide range of voices in American, British, and Brazilian Portuguese accents.
- **Custom Voice Upload** – Upload your own `.pt` model files to use personalized voices.
- **Voice Mixing** – Create unique voices by blending existing ones using weighted formulas.
- **Adjustable Speed** – Control the speech speed from 0.5x to 4x for optimal listening experience.
- **Phoneme Sequence Output** – Get the phoneme sequence corresponding to the generated speech.
- **Efficient Caching System** – Faster processing and voice loading through optimized caching.
- **Easy-to-Use Interface** – An intuitive interface powered by Gradio.

## Usage

Once the app is running, you can interact with the following features:

1. **Input Text** – Type the text you want to convert to speech.
2. **Select Voice** – Choose from a variety of available voices.
3. **Adjust Speed** – Control the speech speed from 0.5x to 4x.
4. **Generate Speech** – Click the "Generate Speech" button to produce the audio. The generated speech will be available for download.
5. **Phoneme Sequence** – View the phoneme sequence that corresponds to the generated speech in the advanced details section.
6. **Custom Voices** – Upload `.pt` files to add new voices to the system.
7. **Voice Mixing** – Combine multiple voices using weight-based formulas to create unique voice profiles.

## Custom Voices

You can upload `.pt` model files to add new voices. The system will automatically integrate them into the voice selection menu. Steps to add a custom voice:

1. Navigate to the **Custom Voices** tab.
2. Enter a unique name for your voice.
3. Upload a `.pt` file containing the voice model.
4. The new voice will be available in the selection dropdown under "👤 Custom Voices."

## Voice Mixing

The voice mixing feature allows users to blend multiple voices using a formula like this:

```
voice1 * 0.7 + voice2 * 0.3
```

This allows you to create unique variations by adjusting the weight distribution. Steps to create a mixed voice:

1. Navigate to the **Voice Mixer** tab.
2. Select the voices you want to blend.
3. Adjust the weight for each voice.
4. Click "Generate Mixed Voice."
5. The new voice will be saved and available in the selection dropdown.

## Performance Optimization

- **GPU Acceleration** – Using a CUDA-compatible GPU significantly improves performance.
- **Efficient Caching** – Models and voices are now cached for faster loading.

## Tips for Better Results

- **Add punctuation** – Helps the model create natural pauses and intonation.
- **Use complete sentences** – The model works best with grammatically correct text.
- **Try different speeds** – Some voices sound better at certain speeds.
- **Match voice to content** – Choose voices that complement your content’s tone.
- **For long texts** – Keep input under 5000 characters per generation for best results.

## License

Kokoro TTS is powered by the Kokoro model and is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

We welcome contributions! Feel free to submit issues or pull requests to improve the model, documentation, or UI.

## Support

For any issues or questions, please open an issue on the GitHub repository or contact support.

---

Happy speech synthesis! Enjoy bringing your words to life with Kokoro TTS. 🎙️

