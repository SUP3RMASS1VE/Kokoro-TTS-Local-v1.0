import os
import random
import torch
import gradio as gr
from datetime import datetime
from kokoro import KModel, KPipeline
from tqdm import tqdm
from scipy.io.wavfile import write

CUDA_AVAILABLE = torch.cuda.is_available()

models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [True]}
if CUDA_AVAILABLE:
    print("Model loaded to GPU.")
else:
    print("Model loaded to CPU.")

pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'abp'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

CHAR_LIMIT = 5000

output_folder = os.path.join(os.getcwd(), 'outputs')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

CHOICES = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇺🇸 🚹 Puck': 'am_puck',
    '🇺🇸 🚹 Echo': 'am_echo',
    '🇺🇸 🚹 Eric': 'am_eric',
    '🇺🇸 🚹 Liam': 'am_liam',
    '🇺🇸 🚹 Onyx': 'am_onyx',
    '🇺🇸 🚹 Santa': 'am_santa',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 Emma': 'bf_emma',
    '🇬🇧 🚺 Isabella': 'bf_isabella',
    '🇬🇧 🚺 Alice': 'bf_alice',
    '🇬🇧 🚺 Lily': 'bf_lily',
    '🇬🇧 🚹 George': 'bm_george',
    '🇬🇧 🚹 Fable': 'bm_fable',
    '🇬🇧 🚹 Lewis': 'bm_lewis',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
    'PF 🚺 Dora': 'pf_dora',
    'PM 🚹 Alex': 'pm_alex',
    'PM 🚹 Santa': 'pm_santa',
}

def preload_voices():
    for voice in CHOICES.values():
        pipeline = pipelines[voice[0]]
        pipeline.load_voice(voice)

preload_voices()

def forward(ps, ref_s, speed):
    try:
        if CUDA_AVAILABLE:
            return models[True](ps, ref_s, speed)
        else:
            return models[False](ps, ref_s, speed)
    except Exception as e:
        print(f"Error with GPU processing: {e}. Falling back to CPU.")
        return models[False](ps, ref_s, speed)

def generate_first(text, voice='af_heart', speed=1):
    text = text.strip()
    
    chunks = [text[i:i + CHAR_LIMIT] for i in range(0, len(text), CHAR_LIMIT)]
    
    audio_output = []
    ps_output = []

    for chunk in tqdm(chunks, desc="Processing chunks", ncols=100):
        pipeline = pipelines[voice[0]]
        pack = pipeline.load_voice(voice)
        
        for _, ps, _ in pipeline(chunk, voice, speed):
            ref_s = pack[len(ps)-1]
            try:
                audio = forward(ps, ref_s, speed)
            except gr.exceptions.Error as e:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU.')
                audio = models[False](ps, ref_s, speed)
            
            audio_output.append(torch.tensor(audio.numpy()))
            ps_output.append(ps)
    
    audio_combined = torch.cat(audio_output, dim=-1)
    
    audio_combined_numpy = audio_combined.detach().cpu().numpy()

    phoneme_sequence = '\n'.join(ps_output)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"audio_{timestamp}.wav"
    audio_filepath = os.path.join(output_folder, audio_filename)
    
    write(audio_filepath, 24000, audio_combined_numpy)

    return audio_filepath, phoneme_sequence

with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown(
            """
            # Kokoro TTS Local
            Welcome to Kokoro, a high-quality text-to-speech synthesis program powered by deep learning. 
            Input any text, choose a voice, adjust the speed, and generate high-fidelity speech in just a few seconds!
            """
        )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Enter Text', placeholder="Type something here...", lines=5)
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Select Voice', info='Choose a voice for the output. Quality and availability vary by language.')
            speed = gr.Slider(minimum=0.5, maximum=4, value=1, step=0.1, label='Speech Speed', info='Adjust the speed of the generated speech (0.5 to 4x).')
        with gr.Column():
            generate_btn = gr.Button('Generate Speech', variant='primary', size='lg')
            out_audio = gr.Audio(label='Generated Speech', interactive=False, streaming=False, autoplay=True)
            out_ps = gr.Textbox(interactive=False, show_label=False, info='Phonemes: The phoneme sequence corresponding to the input text.')

    generate_btn.click(fn=generate_first, inputs=[text, voice, speed], outputs=[out_audio, out_ps])
    
    # Information section at the bottom
    with gr.Row():
        gr.Markdown(
            """
            ## Voice Information:
            - **af/am** = (American English) Female/Male
            - **bf/bm** = (British English) Female/Male
            - **pf/pm** = (Brazilian Portuguese) Female/Male
            """
        )

app.launch()
