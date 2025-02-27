# Kokoro TTS Local

Welcome to Kokoro, a high-quality text-to-speech synthesis program powered by deep learning. This tool converts any text into high-fidelity speech in just a few seconds. Simply input text, select a voice, adjust the speed, and enjoy the generated audio.

## Features

- **High-Quality TTS**: Uses the Kokoro model to produce realistic and natural-sounding speech.
- **Multiple Voices**: Choose from a wide range of voices in American, British, and Brazilian Portuguese accents.
- **Adjustable Speed**: Control the speech speed from 0.5x to 4x for optimal listening experience.
- **Phoneme Sequence Output**: Get the phoneme sequence corresponding to the generated speech.
- **Easy-to-Use Interface**: An intuitive interface powered by Gradio.

## Usage

Once the app is running, you can interact with the following features:

1. **Input Text**: Type the text you want to convert to speech.
2. **Select Voice**: Choose from a variety of voices available in American English, British English, and Brazilian Portuguese.
3. **Adjust Speed**: Control the speech speed from 0.5x to 4x.
4. **Generate Speech**: Click the "Generate Speech" button to produce the audio. The generated speech will be available for download.
5. **Phoneme Sequence**: View the phoneme sequence that corresponds to the generated speech in the advanced details section.

## Voice Guide

The available voices are categorized by language and gender. Below is a list of voices you can choose from:

### 🇺🇸 American English
#### Female Voices
- **Heart ❤️**: Warm, friendly, general-purpose
- **Bella 🔥**: Energetic, enthusiastic
- **Nicole 🎧**: Clear, professional
- **Aoede**: Melodic, poetic
- **Kore**: Calm, relaxation
- **Sarah**: Conversational, casual
- **Nova**: Modern, tech
- **Sky**: Bright, children’s content
- **Alloy**: Neutral, informational
- **Jessica**: Warm, friendly
- **River**: Smooth, audiobooks

#### Male Voices
- **Michael**: Professional, business
- **Fenrir**: Deep, resonant
- **Puck**: Playful, light-hearted
- **Echo**: Clear, instructional
- **Eric**: Authoritative, educational
- **Liam**: Conversational, interviews
- **Onyx**: Rich, narration
- **Santa**: Jolly, holiday content
- **Adam**: Neutral, general-purpose

### 🇬🇧 British English
#### Female Voices
- **Emma**: Refined, formal
- **Isabella**: Elegant, sophisticated
- **Alice**: Clear, proper
- **Lily**: Gentle, storytelling

#### Male Voices
- **George**: Distinguished, formal
- **Fable**: Storyteller, fiction
- **Lewis**: Articulate, educational
- **Daniel**: Conversational, interviews

### 🇧🇷 Brazilian Portuguese
#### Female Voices
- **Dora**: Clear, general-purpose

#### Male Voices
- **Alex**: Professional, business
- **Santa**: Festive, holiday content

## Tips for Better Results

- **Add punctuation**: It helps the model create natural pauses and intonation.
- **Use complete sentences**: The model works best with grammatically correct text.
- **Try different speeds**: Some voices sound better at certain speeds.
- **Match voice to content**: Choose voices that complement your content’s tone.

## Advanced Settings

### Character Limit
The maximum number of characters per generation is **5000**. For larger texts, consider breaking them into smaller chunks for better processing.

### Special Content Handling
- **Numbers**: Spell out numbers as words to ensure correct pronunciation.
- **Acronyms**: Add periods or write them out fully to improve pronunciation.
- **Foreign Words**: The model handles common foreign words but may struggle with uncommon ones.

### Performance Optimization
- **GPU Acceleration**: Using a CUDA-compatible GPU greatly improves performance.
- **CPU Fallback**: If a GPU is unavailable, the app will automatically switch to CPU.

## License

Kokoro TTS is powered by the Kokoro model and is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

If you’d like to contribute to the project, feel free to submit issues or pull requests. We welcome any improvements to the model, documentation, or user interface.

## Support

For any issues or questions, please open an issue on the GitHub repository or contact support.

---

Happy speech synthesis! Enjoy bringing your words to life with Kokoro TTS.
