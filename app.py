import os
import random
import torch
import gradio as gr
from datetime import datetime
from kokoro import KModel, KPipeline
from tqdm import tqdm
from scipy.io.wavfile import write
import warnings

# Set explicit cache directories to ensure consistent caching
cache_base = os.path.abspath(os.path.join(os.getcwd(), 'cache'))
os.environ["HF_HOME"] = os.path.abspath(os.path.join(cache_base, 'HF_HOME'))
os.environ["TORCH_HOME"] = os.path.abspath(os.path.join(cache_base, 'TORCH_HOME'))
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print(f"Using cache directory: {os.environ['HF_HOME']}")

torch.nn.utils.parametrize = torch.nn.utils.parametrizations.weight_norm
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

CUDA_AVAILABLE = torch.cuda.is_available()

# Load models with environment variables controlling cache location
models = {gpu: KModel(repo_id="hexgrad/Kokoro-82M").to('cuda' if gpu else 'cpu').eval() for gpu in [True]}
if CUDA_AVAILABLE:
    print("Model loaded to GPU.")
else:
    print("Model loaded to CPU.")

# Load pipelines with environment variables controlling cache location
pipelines = {lang_code: KPipeline(repo_id="hexgrad/Kokoro-82M", lang_code=lang_code, model=False) for lang_code in 'abp'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

# Store loaded voices to avoid reloading
loaded_voices = {}

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
    print("Preloading voices...")
    for voice_name, voice_id in CHOICES.items():
        print(f"Loading voice: {voice_name} ({voice_id})")
        pipeline = pipelines[voice_id[0]]
        try:
            voice_pack = pipeline.load_voice(voice_id)
            loaded_voices[voice_id] = voice_pack
            print(f"Successfully loaded voice: {voice_name}")
        except Exception as e:
            print(f"Error loading voice {voice_name}: {str(e)}")
    print(f"All voices preloaded successfully. Total voices in cache: {len(loaded_voices)}")

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

    # Use the preloaded voice pack from our cache
    pipeline = pipelines[voice[0]]
    
    # Get voice from in-memory cache
    if voice in loaded_voices:
        pack = loaded_voices[voice]
        print(f"Using cached voice: {voice}")
    else:
        print(f"Voice {voice} not found in cache, loading now...")
        pack = pipeline.load_voice(voice)
        loaded_voices[voice] = pack
    
    for chunk in tqdm(chunks, desc="Processing chunks", ncols=100):
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

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")) as app:
    with gr.Row():
        gr.Markdown(
            """
            # 🎙️ Kokoro TTS Local
            
            Welcome to Kokoro, a high-quality text-to-speech synthesis program powered by deep learning.
            Input any text, choose a voice, adjust the speed, and generate high-fidelity speech in just a few seconds!
            
            <div style="text-align: center; margin: 10px 0;">
                <span style="background: linear-gradient(90deg, #6366F1, #A855F7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.2em; font-weight: bold;">Bringing your words to life with natural-sounding voices</span>
            </div>
            """
        )
    
    with gr.Tabs():
        with gr.TabItem("Generate Speech"):
            with gr.Row():
                with gr.Column(scale=2):
                    text = gr.Textbox(
                        label='Enter Text', 
                        placeholder="Type something here to convert to speech...", 
                        lines=8,
                        elem_id="text-input"
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            voice = gr.Dropdown(
                                list(CHOICES.items()), 
                                value='af_heart', 
                                label='Select Voice', 
                                info='Choose a voice for the output',
                                elem_id="voice-selector"
                            )
                        with gr.Column(scale=1):
                            speed = gr.Slider(
                                minimum=0.5, 
                                maximum=4, 
                                value=1, 
                                step=0.1, 
                                label='Speech Speed', 
                                info='Adjust speed (0.5 to 4x)',
                                elem_id="speed-control"
                            )
                    
                    generate_btn = gr.Button('🔊 Generate Speech', variant='primary', size='lg')
                
                with gr.Column(scale=1):
                    out_audio = gr.Audio(
                        label='Generated Speech', 
                        interactive=False, 
                        streaming=False, 
                        autoplay=True,
                        elem_id="audio-output"
                    )
                    
                    with gr.Accordion("Advanced Details", open=False):
                        out_ps = gr.Textbox(
                            interactive=False, 
                            label="Phoneme Sequence", 
                            info='The phoneme sequence corresponding to the input text',
                            elem_id="phoneme-output"
                        )
        
        with gr.TabItem("Voice Guide"):
            gr.Markdown(
                """
                ## 🎭 Voice Selection Guide
                
                ### 🇺🇸 American English
                
                #### Female Voices
                | Voice | Style | Best For |
                |-------|-------|----------|
                | Heart ❤️ | Warm, friendly | General purpose, narration |
                | Bella 🔥 | Energetic | Marketing, enthusiastic content |
                | Nicole 🎧 | Clear, professional | Tutorials, business |
                | Aoede | Melodic | Storytelling, poetry |
                | Kore | Calm | Meditation, relaxation content |
                | Sarah | Conversational | Dialogue, casual content |
                | Nova | Modern | Tech content, news |
                | Sky | Bright | Children's content, upbeat messages |
                | Alloy | Neutral | Documentation, informational |
                | Jessica | Warm | Friendly conversations |
                | River | Smooth | Audiobooks, long-form content |
                
                #### Male Voices
                | Voice | Style | Best For |
                |-------|-------|----------|
                | Michael | Professional | Business, formal content |
                | Fenrir | Deep, resonant | Dramatic readings |
                | Puck | Playful | Light-hearted content |
                | Echo | Clear | Instructional content |
                | Eric | Authoritative | Educational material |
                | Liam | Conversational | Dialogue, interviews |
                | Onyx | Rich | Narration, documentaries |
                | Santa | Jolly | Holiday content, children's stories |
                | Adam | Neutral | General purpose |
                
                ### 🇬🇧 British English
                
                #### Female Voices
                | Voice | Style | Best For |
                |-------|-------|----------|
                | Emma | Refined | Formal content, literature |
                | Isabella | Elegant | Sophisticated content |
                | Alice | Clear, proper | Educational, instructional |
                | Lily | Gentle | Storytelling, children's content |
                
                #### Male Voices
                | Voice | Style | Best For |
                |-------|-------|----------|
                | George | Distinguished | Formal presentations, documentaries |
                | Fable | Storyteller | Narratives, fiction |
                | Lewis | Articulate | Educational content |
                | Daniel | Conversational | Dialogue, interviews |
                
                ### 🇧🇷 Brazilian Portuguese
                
                | Voice | Style | Best For |
                |-------|-------|----------|
                | Dora (F) | Clear | General purpose |
                | Alex (M) | Professional | Business, formal content |
                | Santa (M) | Festive | Holiday content |
                """
            )
        
        with gr.TabItem("Tips & Tricks"):
            gr.Markdown(
                """
                ## 💡 Tips for Better Results
                
                ### Improve Speech Quality
                
                - **Add punctuation**: Proper punctuation helps create natural pauses and intonation
                - **Use complete sentences**: The model performs better with grammatically complete phrases
                - **Try different speeds**: Some voices sound more natural at slightly faster or slower speeds
                - **Consider voice-content match**: Choose voices that match the tone of your content
                
                ### Handling Special Content
                
                - **Numbers**: Write out numbers as words for better pronunciation of important figures
                - **Acronyms**: Add periods between letters (like "U.S.A.") or write them out
                - **Foreign words**: The model handles common foreign words, but may struggle with uncommon ones
                - **Technical terms**: For domain-specific terminology, test different voices
                
                ### Performance Tips
                
                - **Character limit**: Remember there's a ${CHAR_LIMIT} character limit per generation
                - **For longer texts**: Break into smaller chunks for better processing
                - **GPU acceleration**: Using a GPU significantly improves generation speed
                """
            )

    with gr.Row(variant="panel"):
        gr.Markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3>Voice Code Legend:</h3>
                    <span class="voice-code">af/am</span> = American English (Female/Male) &nbsp;|&nbsp;
                    <span class="voice-code">bf/bm</span> = British English (Female/Male) &nbsp;|&nbsp;
                    <span class="voice-code">pf/pm</span> = Brazilian Portuguese (Female/Male)
                </div>
                <div>
                    <p style="text-align: right; font-style: italic; margin: 0;">Powered by Kokoro TTS</p>
                </div>
            </div>
            
            <style>
                .voice-code {
                    font-family: monospace;
                    background-color: #6366F1;
                    color: white;
                    padding: 3px 6px;
                    border-radius: 4px;
                    font-weight: bold;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                }
            </style>
            """
        )

    generate_btn.click(fn=generate_first, inputs=[text, voice, speed], outputs=[out_audio, out_ps])
    
    # Add custom CSS for better styling
    app.load(js="""
    function() {
        // Add custom styling
        const style = document.createElement('style');
        style.textContent = `
            /* Custom button hover effect */
            button.primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
                transition: all 0.3s ease;
            }
            
            /* Improve dropdown styling */
            #voice-selector select {
                border-radius: 8px;
            }
            
            /* Add subtle animation to the generate button */
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
                100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
            }
            
            button.primary {
                animation: pulse 2s infinite;
            }
            
            /* Improve text input area */
            #text-input textarea {
                border-radius: 8px;
                transition: border-color 0.3s ease;
            }
            
            #text-input textarea:focus {
                border-color: #6366F1;
                box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
            }
        `;
        document.head.appendChild(style);
    }
    """)

app.launch()
