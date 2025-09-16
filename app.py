import os
import random
import torch
import gradio as gr
import shutil
from datetime import datetime
from kokoro import KModel, KPipeline
from tqdm import tqdm
from scipy.io.wavfile import write, read
import subprocess
import warnings

# Set explicit cache directories to ensure consistent caching
cache_base = os.path.abspath(os.path.join(os.getcwd(), 'cache'))
os.environ["HF_HOME"] = os.path.abspath(os.path.join(cache_base, 'HF_HOME'))
os.environ["TORCH_HOME"] = os.path.abspath(os.path.join(cache_base, 'TORCH_HOME'))
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Set Gradio temp directory to our outputs folder to avoid duplicate file storage
output_folder = os.path.join(os.getcwd(), 'outputs')
os.environ["GRADIO_TEMP_DIR"] = os.path.abspath(output_folder)
# Add these environment variables to prevent redownloading models each time
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

print(f"Using cache directory: {os.environ['HF_HOME']}")

torch.nn.utils.parametrize = torch.nn.utils.parametrizations.weight_norm
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

CUDA_AVAILABLE = torch.cuda.is_available()

try:
    # First run - download models if they don't exist
    if not os.path.exists(os.path.join(cache_base, 'HF_HOME/hub/models--hexgrad--Kokoro-82M')):
        print("First run detected, downloading models...")
        # Temporarily disable offline mode to allow downloads
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)
        
    # Load models with environment variables controlling cache location
    models = {gpu: KModel(repo_id="hexgrad/Kokoro-82M").to('cuda' if gpu else 'cpu').eval() for gpu in [True]}
    if CUDA_AVAILABLE:
        print("Model loaded to GPU.")
    else:
        print("Model loaded to CPU.")

    # Load pipelines with environment variables controlling cache location
    pipelines = {lang_code: KPipeline(repo_id="hexgrad/Kokoro-82M", lang_code=lang_code, model=False) for lang_code in 'abpi'}
    pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
    pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
    # Add try-except for Italian pipeline which might not have lexicon attribute
    try:
        if hasattr(pipelines['i'].g2p, 'lexicon'):
            pipelines['i'].g2p.lexicon.golds['kokoro'] = 'kˈkɔro'
        else:
            print("Warning: Italian pipeline g2p doesn't have lexicon attribute, skipping custom pronunciation")
    except Exception as e:
        print(f"Warning: Could not set custom pronunciation for Italian: {str(e)}")
    
    # After successful loading, re-enable offline mode to prevent future download attempts
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
except Exception as e:
    print(f"Error during model loading: {str(e)}")
    print("Attempting to load in online mode...")
    # If offline loading fails, try online mode
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    
    # Load models with environment variables controlling cache location
    models = {gpu: KModel(repo_id="hexgrad/Kokoro-82M").to('cuda' if gpu else 'cpu').eval() for gpu in [True]}
    if CUDA_AVAILABLE:
        print("Model loaded to GPU.")
    else:
        print("Model loaded to CPU.")

    # Load pipelines with environment variables controlling cache location
    pipelines = {lang_code: KPipeline(repo_id="hexgrad/Kokoro-82M", lang_code=lang_code, model=False) for lang_code in 'abpi'}
    pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
    pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
    # Add try-except for Italian pipeline which might not have lexicon attribute
    try:
        if hasattr(pipelines['i'].g2p, 'lexicon'):
            pipelines['i'].g2p.lexicon.golds['kokoro'] = 'kˈkɔro'
        else:
            print("Warning: Italian pipeline g2p doesn't have lexicon attribute, skipping custom pronunciation")
    except Exception as e:
        print(f"Warning: Could not set custom pronunciation for Italian: {str(e)}")

# Store loaded voices to avoid reloading
loaded_voices = {}

CHAR_LIMIT = 5000

custom_voices_folder = os.path.join(os.getcwd(), 'custom_voices')

# Create output folder if it doesn't exist (already defined above for GRADIO_TEMP_DIR)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(custom_voices_folder):
    os.makedirs(custom_voices_folder)

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
    '🇮🇹 🚺 Sara': 'if_sara',
    '🇮🇹 🚹 Nicola': 'im_nicola',
}

# Function to get custom voices from the custom_voices folder
def get_custom_voices():
    custom_voices = {}
    if os.path.exists(custom_voices_folder):
        for file in os.listdir(custom_voices_folder):
            file_path = os.path.join(custom_voices_folder, file)
            # Check if it's a .pt file (PyTorch model file)
            if file.endswith('.pt') and os.path.isfile(file_path):
                voice_id = os.path.splitext(file)[0]  # Remove the .pt extension
                custom_voices[f'👤 Custom: {voice_id}'] = f'custom_{voice_id}'
    return custom_voices

# Update choices with custom voices
def update_voice_choices():
    updated_choices = CHOICES.copy()
    custom_voices = get_custom_voices()
    updated_choices.update(custom_voices)
    return updated_choices

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
    
    # Load custom voices if any
    custom_voices = get_custom_voices()
    for voice_name, voice_id in custom_voices.items():
        try:
            # Custom voices use the American English pipeline by default
            pipeline = pipelines['a']
            voice_file = f"{voice_id.split('_')[1]}.pt"
            voice_path = os.path.join(custom_voices_folder, voice_file)
            
            # Check if the file exists
            if not os.path.exists(voice_path):
                print(f"Custom voice file not found: {voice_file}")
                continue
            
            # Load the .pt file directly
            try:
                voice_pack = torch.load(voice_path, weights_only=True)
                loaded_voices[voice_id] = voice_pack
                print(f"Successfully loaded custom voice: {voice_name}")
            except Exception as e:
                print(f"Error loading custom voice {voice_name}: {str(e)}")
        except Exception as e:
            print(f"Error loading custom voice {voice_name}: {str(e)}")
    
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

def convert_to_mp3(input_wav_path, output_mp3_path, bitrate="192k"):
    """Convert WAV file to MP3 using ffmpeg"""
    try:
        # Import ffmpeg from imageio
        import imageio_ffmpeg as ffmpeg
        
        # Get ffmpeg executable path
        ffmpeg_path = ffmpeg.get_ffmpeg_exe()
        
        # Get input file size for progress info
        input_size_mb = os.path.getsize(input_wav_path) / (1024 * 1024)
        print(f"🔄 Converting {input_size_mb:.1f} MB WAV to MP3 (bitrate: {bitrate})...")
        
        # Build ffmpeg command
        cmd = [
            ffmpeg_path,
            '-i', input_wav_path,
            '-codec:a', 'libmp3lame',
            '-b:a', bitrate,
            '-y',  # Overwrite output file if it exists
            output_mp3_path
        ]
        
        print(f"⚙️  Running FFmpeg conversion...")
        # Run ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ MP3 conversion completed successfully!")
            return True
        else:
            print(f"❌ FFmpeg conversion failed!")
            print(f"Error details: {result.stderr}")
            return False
            
    except ImportError:
        print("❌ imageio-ffmpeg not available. Please install it with: pip install imageio-ffmpeg")
        return False
    except Exception as e:
        print(f"❌ Error during MP3 conversion: {str(e)}")
        return False

def generate_first(text, voice='af_heart', speed=1, output_format='WAV'):
    text = text.strip()
    
    # Check if the voice is a display name from standard voices
    if voice in CHOICES:
        voice = CHOICES[voice]
    # Check if the voice is a custom voice display name
    elif voice.startswith('👤 Custom:'):
        custom_voices = get_custom_voices()
        if voice in custom_voices:
            voice = custom_voices[voice]
        else:
            raise gr.Error(f"Custom voice not found: {voice}")
    
    chunks = [text[i:i + CHAR_LIMIT] for i in range(0, len(text), CHAR_LIMIT)]
    
    audio_output = []
    ps_output = []

    # Determine if this is a custom voice
    is_custom = voice.startswith('custom_')
    
    # Use the appropriate pipeline
    if is_custom:
        pipeline = pipelines['a']  # Use American English pipeline for custom voices
    else:
        pipeline = pipelines[voice[0]]
    
    # Get voice from in-memory cache or load it
    if voice in loaded_voices:
        pack = loaded_voices[voice]
        print(f"Using cached voice: {voice}")
    else:
        print(f"Voice {voice} not found in cache, loading now...")
        if is_custom:
            # Load custom voice from the custom_voices folder
            voice_name = voice.split('_')[1]
            voice_file = f"{voice_name}.pt"
            voice_path = os.path.join(custom_voices_folder, voice_file)
            
            # Check if the file exists
            if not os.path.exists(voice_path):
                raise gr.Error(f"Custom voice file not found: {voice_file}")
            
            # Load the .pt file directly
            try:
                pack = torch.load(voice_path, weights_only=True)
            except Exception as e:
                raise gr.Error(f"Error loading custom voice: {str(e)}")
        else:
            pack = pipeline.load_voice(voice)
        loaded_voices[voice] = pack
    
    for chunk in tqdm(chunks, desc="Processing chunks", ncols=100):
        for _, ps, _ in pipeline(chunk, voice if not is_custom else None, speed):
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
    
    # Calculate file size information
    audio_length_seconds = len(audio_combined_numpy) / 24000
    estimated_wav_size_mb = (len(audio_combined_numpy) * 2) / (1024 * 1024)  # 16-bit audio
    
    print(f"Audio generation complete!")
    print(f"Audio length: {audio_length_seconds:.1f} seconds ({audio_length_seconds/60:.1f} minutes)")
    print(f"Estimated WAV file size: {estimated_wav_size_mb:.1f} MB")
    
    # Handle different output formats
    if output_format.upper() == 'MP3':
        # Save as WAV first, then convert to MP3
        wav_filename = f"audio_{timestamp}.wav"
        wav_filepath = os.path.join(output_folder, wav_filename)
        
        print(f"Saving audio as WAV file: {wav_filename}")
        write(wav_filepath, 24000, audio_combined_numpy)
        actual_wav_size_mb = os.path.getsize(wav_filepath) / (1024 * 1024)
        print(f"WAV file saved successfully! Actual size: {actual_wav_size_mb:.1f} MB")
        
        # Convert to MP3
        audio_filename = f"audio_{timestamp}.mp3"
        audio_filepath = os.path.join(output_folder, audio_filename)
        
        print(f"Starting MP3 conversion...")
        print(f"Converting: {wav_filename} → {audio_filename}")
        
        # Use ffmpeg for conversion
        if convert_to_mp3(wav_filepath, audio_filepath):
            # Check MP3 file size
            mp3_size_mb = os.path.getsize(audio_filepath) / (1024 * 1024)
            compression_ratio = (actual_wav_size_mb / mp3_size_mb) if mp3_size_mb > 0 else 0
            print(f"MP3 conversion successful!")
            print(f"MP3 file size: {mp3_size_mb:.1f} MB (compression ratio: {compression_ratio:.1f}x)")
            
            # Try to remove the WAV file after successful conversion
            try:
                os.remove(wav_filepath)
                print(f"Temporary WAV file removed: {wav_filename}")
                print(f"Final output: {audio_filename}")
            except PermissionError:
                print(f"Warning: Could not delete WAV file (file in use): {wav_filename}")
                print("The MP3 conversion was successful. You can manually delete the WAV file later.")
            except Exception as e:
                print(f"Warning: Could not delete WAV file: {str(e)}")
        else:
            # If MP3 conversion fails, keep the WAV file and return it
            print("MP3 conversion failed. Keeping WAV format.")
            audio_filename = wav_filename
            audio_filepath = wav_filepath
    else:
        # Default WAV format
        audio_filename = f"audio_{timestamp}.wav"
        audio_filepath = os.path.join(output_folder, audio_filename)
        
        print(f"Saving audio as WAV file: {audio_filename}")
        write(audio_filepath, 24000, audio_combined_numpy)
        actual_wav_size_mb = os.path.getsize(audio_filepath) / (1024 * 1024)
        print(f"WAV file saved successfully! Size: {actual_wav_size_mb:.1f} MB")

    # Check if file is too large for proper waveform display
    final_file_size_mb = os.path.getsize(audio_filepath) / (1024 * 1024)
    is_large_file = final_file_size_mb > 50  # Consider files over 50MB as large
    
    if is_large_file:
        print(f"⚠️  Large file generated ({final_file_size_mb:.1f} MB)")
        print(f"📁 File location: {audio_filepath}")
        print(f"💡 Note: Large files may not display waveforms properly in the browser.")
        print(f"   You can access the full file directly from the outputs folder.")
    
    print(f"🎵 Generation complete! Total processing time for {len(chunks)} chunks.")
    
    return audio_filepath, phoneme_sequence, gr.update(visible=is_large_file)

# Function to handle custom voice upload
def upload_custom_voice(files, voice_name):
    if not voice_name or not voice_name.strip():
        return "Please provide a name for your custom voice."
    
    # Sanitize voice name (remove spaces and special characters)
    voice_name = ''.join(c for c in voice_name if c.isalnum() or c == '_')
    
    if not voice_name:
        return "Invalid voice name. Please use alphanumeric characters."
    
    # Check if any files were uploaded
    if not files:
        return "Please upload a .pt voice file."
    
    # In Gradio, the file object structure depends on the file_count parameter
    # For file_count="single", files is the file path as a string
    file_path = files
    
    # Check if the uploaded file is a .pt file
    if not file_path.endswith('.pt'):
        return "Please upload a valid .pt voice file."
    
    # Copy the file to the custom_voices folder with the new name
    target_file = os.path.join(custom_voices_folder, f"{voice_name}.pt")
    
    # If file already exists, remove it
    if os.path.exists(target_file):
        os.remove(target_file)
    
    # Copy the uploaded file
    shutil.copy(file_path, target_file)
    
    # Try to load the voice to verify it works
    voice_id = f'custom_{voice_name}'
    
    try:
        # Load the .pt file directly
        voice_pack = torch.load(target_file, weights_only=True)
        
        # Verify that the voice pack is usable with the model
        # Check if it's a tensor or a list/tuple of tensors
        if not isinstance(voice_pack, (torch.Tensor, list, tuple)):
            raise ValueError("The voice file is not in the expected format (should be a tensor or list of tensors)")
        
        # If it's a list or tuple, check that it contains tensors
        if isinstance(voice_pack, (list, tuple)) and (len(voice_pack) == 0 or not isinstance(voice_pack[0], torch.Tensor)):
            raise ValueError("The voice file does not contain valid tensor data")
            
        loaded_voices[voice_id] = voice_pack
        return f"Custom voice '{voice_name}' uploaded and loaded successfully!"
    except Exception as e:
        # If loading fails, remove the file
        if os.path.exists(target_file):
            os.remove(target_file)
        return f"Error loading custom voice: {str(e)}"

# Function to handle custom voice upload and refresh lists
def upload_and_refresh(files, voice_name):
    result = upload_custom_voice(files, voice_name)
    
    # If upload was successful, clear the input fields
    if "successfully" in result:
        return result, get_custom_voice_list(), "", None
    else:
        return result, get_custom_voice_list(), voice_name, files

# Function to refresh the voice list
def refresh_voice_list():
    updated_choices = update_voice_choices()
    return gr.update(choices=list(updated_choices.keys()), value=list(updated_choices.keys())[0])

# Function to get the list of custom voices for the dataframe
def get_custom_voice_list():
    custom_voices = get_custom_voices()
    if not custom_voices:
        return [["No custom voices found", "N/A"]]
    return [[name.replace('👤 Custom: ', ''), "Loaded"] for name in custom_voices.keys()]

# Add voice mixing functionality
def parse_voice_formula(formula):
    if not formula.strip():
        raise ValueError("Empty voice formula")
    
    weighted_sum = None
    terms = formula.split('+')
    weights = 0
    
    for term in terms:
        parts = term.strip().split('*')
        if len(parts) != 2:
            raise ValueError(f"Invalid term format: {term.strip()}")
        
        voice_name = parts[0].strip()
        weight = float(parts[1].strip())
        weights += weight
        
        if voice_name not in loaded_voices:
            raise ValueError(f"Unknown voice: {voice_name}")
        
        voice_tensor = loaded_voices[voice_name]
        
        if weighted_sum is None:
            weighted_sum = weight * voice_tensor
        else:
            weighted_sum += weight * voice_tensor
    
    return weighted_sum / weights

def get_new_voice(formula, custom_name=""):
    try:
        weighted_voices = parse_voice_formula(formula)
        
        # Create a filename with custom name or timestamp if no name provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_name and custom_name.strip():
            # Sanitize custom name (remove spaces and special characters)
            custom_name = ''.join(c for c in custom_name if c.isalnum() or c == '_')
            voice_name = f"{custom_name}"
        else:
            voice_name = f"mixed_{timestamp}"
            
        voice_pack_name = os.path.join(custom_voices_folder, f"{voice_name}.pt")
        
        torch.save(weighted_voices, voice_pack_name)
        return voice_pack_name, voice_name
    except Exception as e:
        raise gr.Error(f"Failed to create voice: {str(e)}")

def generate_mixed_voice(formula_text, voice_name="", text_input=""):
    try:
        # Create the mixed voice file with custom name
        voice_file_path, voice_name = get_new_voice(formula_text, voice_name)
        voice_id = f"custom_{voice_name}"
        
        # Load the voice into memory to ensure it's available
        voice_pack = torch.load(voice_file_path, weights_only=True)
        loaded_voices[voice_id] = voice_pack
        
        # If text input is provided, generate audio with the mixed voice
        if text_input.strip():
            audio_path, _, _ = generate_first(text_input, voice_id)
            return f"Mixed voice '{voice_name}' created successfully! You can now select it from the voice dropdown as '👤 Custom: {voice_name}'", audio_path
        else:
            return f"Mixed voice '{voice_name}' created successfully! You can now select it from the voice dropdown as '👤 Custom: {voice_name}'", None
    except Exception as e:
        raise gr.Error(f"Failed to generate mixed voice: {e}")

# Function to build voice formula from sliders
def build_formula_from_sliders(*args):
    # The args will contain alternating checkbox and slider values
    formula_parts = []
    
    # Get the organized list of voices in the same order as they appear in the UI
    voice_keys = list(CHOICES.keys())
    voice_keys.sort()
    us_female_voices = [k for k in voice_keys if k.startswith('🇺🇸 🚺')]
    us_male_voices = [k for k in voice_keys if k.startswith('🇺🇸 🚹')]
    gb_female_voices = [k for k in voice_keys if k.startswith('🇬🇧 🚺')]
    gb_male_voices = [k for k in voice_keys if k.startswith('🇬🇧 🚹')]
    other_voices = [k for k in voice_keys if not (k.startswith('🇺🇸') or k.startswith('🇬🇧'))]
    organized_voices = us_female_voices + us_male_voices + gb_female_voices + gb_male_voices + other_voices
    
    for i in range(0, len(args), 2):
        if i+1 < len(args):  # Make sure we have both checkbox and slider
            checkbox = args[i]
            slider = args[i+1]
            
            if checkbox and i//2 < len(organized_voices):  # If checkbox is checked
                voice_name = organized_voices[i//2]
                voice_id = CHOICES[voice_name]
                formula_parts.append(f"{voice_id} * {slider}")
    
    if not formula_parts:
        return ""
    
    return " + ".join(formula_parts)

# Helper function to generate audio without saving to disk
def generate_audio_in_memory(text, voice, speed=1):
    """Generate audio without saving intermediate files"""
    text = text.strip()
    
    # Check if the voice is a display name from standard voices
    if voice in CHOICES:
        voice = CHOICES[voice]
    # Check if the voice is a custom voice display name
    elif voice.startswith('👤 Custom:'):
        custom_voices = get_custom_voices()
        if voice in custom_voices:
            voice = custom_voices[voice]
        else:
            raise gr.Error(f"Custom voice not found: {voice}")
    
    chunks = [text[i:i + CHAR_LIMIT] for i in range(0, len(text), CHAR_LIMIT)]
    
    audio_output = []

    # Determine if this is a custom voice
    is_custom = voice.startswith('custom_')
    
    # Use the appropriate pipeline
    if is_custom:
        pipeline = pipelines['a']  # Use American English pipeline for custom voices
    else:
        pipeline = pipelines[voice[0]]
    
    # Get voice from in-memory cache or load it
    if voice in loaded_voices:
        pack = loaded_voices[voice]
    else:
        if is_custom:
            # Load custom voice from the custom_voices folder
            voice_name = voice.split('_')[1]
            voice_file = f"{voice_name}.pt"
            voice_path = os.path.join(custom_voices_folder, voice_file)
            
            # Check if the file exists
            if not os.path.exists(voice_path):
                raise gr.Error(f"Custom voice file not found: {voice_file}")
            
            # Load the .pt file directly
            try:
                pack = torch.load(voice_path, weights_only=True)
            except Exception as e:
                raise gr.Error(f"Error loading custom voice: {str(e)}")
        else:
            pack = pipeline.load_voice(voice)
        loaded_voices[voice] = pack
    
    for chunk in chunks:
        for _, ps, _ in pipeline(chunk, voice if not is_custom else None, speed):
            ref_s = pack[len(ps)-1]
            try:
                audio = forward(ps, ref_s, speed)
            except gr.exceptions.Error as e:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU.')
                audio = models[False](ps, ref_s, speed)
            
            audio_output.append(torch.tensor(audio.numpy()))
    
    # Return combined audio as tensor
    audio_combined = torch.cat(audio_output, dim=-1)
    return audio_combined

# Function to parse conversation script
def parse_conversation_script(script_text):
    """Parse a conversation script and extract speakers and their lines"""
    if not script_text.strip():
        return []
    
    lines = script_text.strip().split('\n')
    conversation = []
    current_speaker = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with "Speaker X:" pattern
        if ':' in line:
            # Check if it's a speaker line
            potential_speaker = line.split(':', 1)[0].strip()
            if potential_speaker.lower().startswith('speaker') or len(potential_speaker.split()) <= 3:
                # Save previous speaker's text if any
                if current_speaker and current_text:
                    conversation.append((current_speaker, ' '.join(current_text)))
                
                # Start new speaker
                current_speaker = potential_speaker
                current_text = [line.split(':', 1)[1].strip()]
            else:
                # Not a speaker line, add to current text
                if current_speaker:
                    current_text.append(line)
        else:
            # Continuation of current speaker's text
            if current_speaker:
                current_text.append(line)
    
    # Add the last speaker's text
    if current_speaker and current_text:
        conversation.append((current_speaker, ' '.join(current_text)))
    
    return conversation

def trim_silence(audio_tensor, threshold=0.01):
    """Trim silence from the beginning and end of audio"""
    # Find first and last non-silent samples
    non_silent = torch.abs(audio_tensor) > threshold
    if not torch.any(non_silent):
        return audio_tensor  # Return original if all silent
    
    # Find first and last non-silent indices
    first_sound = torch.where(non_silent)[0][0]
    last_sound = torch.where(non_silent)[0][-1]
    
    # Trim with small padding to avoid cutting off audio
    padding = int(24000 * 0.05)  # 50ms padding
    start = max(0, first_sound - padding)
    end = min(len(audio_tensor), last_sound + padding)
    
    return audio_tensor[start:end]

def batch_convert_text_files_with_voices(files, speed, output_format, *voice_assignments):
    """Convert multiple text files to audio using individual voice settings for each file"""
    if not files:
        raise gr.Error("Please upload at least one text file.")
    
    results = []
    audio_files = []  # Store paths to generated audio files
    total_files = len(files)
    
    print(f"Starting batch conversion of {total_files} files...")
    
    for i, file_path in enumerate(files):
        try:
            print(f"Processing file {i+1}/{total_files}: {os.path.basename(file_path)}")
            
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            if not text_content:
                print(f"Skipping empty file: {os.path.basename(file_path)}")
                results.append(f"❌ {os.path.basename(file_path)}: Empty file")
                audio_files.append(None)
                continue
            
            # Get the voice for this specific file
            voice = voice_assignments[i] if i < len(voice_assignments) and voice_assignments[i] else list(update_voice_choices().keys())[0]
            
            # Generate audio for this text with the assigned voice
            audio_path, _, _ = generate_first(text_content, voice, speed, output_format)
            
            # Rename the output file to match the input filename
            input_filename = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format.upper() == 'MP3':
                new_filename = f"{input_filename}_{timestamp}.mp3"
            else:
                new_filename = f"{input_filename}_{timestamp}.wav"
            
            new_audio_path = os.path.join(output_folder, new_filename)
            
            # Rename the generated file
            if os.path.exists(audio_path):
                os.rename(audio_path, new_audio_path)
                file_size_mb = os.path.getsize(new_audio_path) / (1024 * 1024)
                
                # Get voice display name for results
                voice_display = voice if voice in update_voice_choices() else voice
                results.append(f"✅ {os.path.basename(file_path)} → {new_filename} ({file_size_mb:.1f} MB) [Voice: {voice_display}]")
                audio_files.append(new_audio_path)
                print(f"✅ Completed: {new_filename} with voice: {voice}")
            else:
                results.append(f"❌ {os.path.basename(file_path)}: Audio generation failed")
                audio_files.append(None)
                
        except Exception as e:
            error_msg = f"❌ {os.path.basename(file_path)}: {str(e)}"
            results.append(error_msg)
            audio_files.append(None)
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
    
    # Create summary
    successful = len([r for r in results if r.startswith("✅")])
    failed = len([r for r in results if r.startswith("❌")])
    
    summary = f"Batch conversion completed!\n"
    summary += f"✅ Successful: {successful}/{total_files}\n"
    summary += f"❌ Failed: {failed}/{total_files}\n\n"
    summary += "Results:\n" + "\n".join(results)
    
    print(f"Batch conversion completed: {successful} successful, {failed} failed")
    
    return summary, audio_files

def update_file_voice_assignments(files):
    """Update the voice assignment interface when files are uploaded"""
    if not files:
        # Hide all voice assignments if no files
        updates = [gr.update(visible=False)]  # Hide the container
        for i in range(20):
            updates.append(gr.update(visible=False))
        return updates
    
    # Show the container and update voice assignments for each file
    updates = [gr.update(visible=True)]  # Show the container
    
    for i in range(20):
        if i < len(files):
            filename = os.path.basename(files[i])
            updates.append(gr.update(
                visible=True,
                label=f"🎵 Voice for: {filename}",
                value=list(update_voice_choices().keys())[0]  # Default to first voice
            ))
        else:
            updates.append(gr.update(visible=False))
    
    return updates

def assign_same_voice_to_all(voice_choice, files):
    """Assign the same voice to all uploaded files"""
    if not files:
        return [gr.update() for _ in range(20)]
    
    updates = []
    for i in range(20):
        if i < len(files):
            updates.append(gr.update(value=voice_choice))
        else:
            updates.append(gr.update())
    
    return updates

def update_batch_audio_players(audio_files):
    """Update the batch audio players with generated files"""
    updates = []
    
    for i in range(20):  # Match the number of audio players created
        if i < len(audio_files) and audio_files[i] is not None:
            # Show this audio player with the generated file
            filename = os.path.basename(audio_files[i])
            updates.append(gr.update(value=audio_files[i], visible=True, label=f"🎵 {filename}"))
        else:
            # Hide this audio player
            updates.append(gr.update(visible=False))
    
    # Show the audio files column if we have any files
    has_files = any(f is not None for f in audio_files) if audio_files else False
    audio_column_update = gr.update(visible=has_files)
    
    return [audio_column_update] + updates

def generate_conversation_from_script(script_text, speaker_voices, pause_duration, default_speed, output_format='WAV'):
    """Generate conversation audio from a script with assigned voices"""
    conversation = parse_conversation_script(script_text)
    
    if not conversation:
        raise gr.Error("No conversation found. Please enter a script in the format:\nSpeaker 1: Hello\nSpeaker 2: Hi there")
    
    # Get unique speakers
    speakers = list(set([speaker for speaker, _ in conversation]))
    
    # Check if all speakers have assigned voices
    missing_voices = [speaker for speaker in speakers if speaker not in speaker_voices or not speaker_voices[speaker]]
    if missing_voices:
        raise gr.Error(f"Please assign voices for: {', '.join(missing_voices)}")
    
    audio_segments = []
    conversation_script = []
    
    for i, (speaker, text) in enumerate(conversation):
        if not text.strip():
            continue
            
        # Update conversation script
        conversation_script.append(f"{speaker}: {text}")
        
        # Get voice for this speaker
        voice = speaker_voices.get(speaker)
        if not voice:
            continue
            
        # Debug: Print voice information
        print(f"Processing speaker '{speaker}' with voice '{voice}'")
        if voice.startswith('👤 Custom:'):
            custom_voice_name = voice.replace('👤 Custom: ', '')
            custom_voice_file = f"{custom_voice_name}.pt"
            custom_voice_path = os.path.join(custom_voices_folder, custom_voice_file)
            print(f"Custom voice file path: {custom_voice_path}")
            print(f"File exists: {os.path.exists(custom_voice_path)}")
            if not os.path.exists(custom_voice_path):
                # List available custom voice files
                available_files = [f for f in os.listdir(custom_voices_folder) if f.endswith('.pt')] if os.path.exists(custom_voices_folder) else []
                raise gr.Error(f"Custom voice file '{custom_voice_file}' not found in custom_voices folder.\nAvailable custom voice files: {available_files}")
            
        # Generate audio for this speaker in memory (no intermediate files saved)
        try:
            audio_tensor = generate_audio_in_memory(text, voice, default_speed)
            
            # Normalize audio
            if audio_tensor.max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.max()
            
            # Trim silence from individual audio clips
            audio_tensor = trim_silence(audio_tensor)
            
            audio_segments.append(audio_tensor)
            
            # Handle pause between speakers (can be negative for overlap)
            if i < len(conversation) - 1:
                if pause_duration > 0:
                    # Add silence
                    pause_samples = int(24000 * pause_duration)
                    pause_audio = torch.zeros(pause_samples)
                    audio_segments.append(pause_audio)
                elif pause_duration < 0:
                    # Negative pause means trim from the end of current audio
                    trim_samples = int(24000 * abs(pause_duration))
                    if len(audio_segments[-1]) > trim_samples:
                        audio_segments[-1] = audio_segments[-1][:-trim_samples]
                # If pause_duration == 0, add no pause (direct concatenation)
                
        except Exception as e:
            raise gr.Error(f"Error generating audio for {speaker}: {str(e)}")
    
    # Combine all audio segments
    if audio_segments:
        combined_audio = torch.cat(audio_segments, dim=-1)
        combined_audio_numpy = combined_audio.detach().cpu().numpy()
        
        # Save the combined conversation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Handle different output formats
        if output_format.upper() == 'MP3':
            # Save as WAV first, then convert to MP3
            wav_filename = f"conversation_{timestamp}.wav"
            wav_filepath = os.path.join(output_folder, wav_filename)
            write(wav_filepath, 24000, combined_audio_numpy)
            
            # Convert to MP3
            conversation_filename = f"conversation_{timestamp}.mp3"
            conversation_filepath = os.path.join(output_folder, conversation_filename)
            
            # Use ffmpeg for conversion
            if convert_to_mp3(wav_filepath, conversation_filepath):
                # Check MP3 file size
                mp3_size_mb = os.path.getsize(conversation_filepath) / (1024 * 1024)
                wav_size_mb = os.path.getsize(wav_filepath) / (1024 * 1024)
                compression_ratio = (wav_size_mb / mp3_size_mb) if mp3_size_mb > 0 else 0
                print(f"MP3 conversion successful!")
                print(f"MP3 file size: {mp3_size_mb:.1f} MB (compression ratio: {compression_ratio:.1f}x)")
                
                # Try to remove the WAV file after successful conversion
                try:
                    os.remove(wav_filepath)
                    print(f"Temporary WAV file removed: {wav_filename}")
                    print(f"Final output: {conversation_filename}")
                except PermissionError:
                    print(f"Warning: Could not delete WAV file (file in use): {wav_filename}")
                    print("The MP3 conversion was successful. You can manually delete the WAV file later.")
                except Exception as e:
                    print(f"Warning: Could not delete WAV file: {str(e)}")
            else:
                # If MP3 conversion fails, keep the WAV file and return it
                print("MP3 conversion failed. Keeping WAV format.")
                conversation_filename = wav_filename
                conversation_filepath = wav_filepath
        else:
            # Default WAV format
            conversation_filename = f"conversation_{timestamp}.wav"
            conversation_filepath = os.path.join(output_folder, conversation_filename)
            write(conversation_filepath, 24000, combined_audio_numpy)
        
        # Create conversation script text
        script_text = "\n".join(conversation_script)
        
        print(f"🎬 Conversation generation complete!")
        print(f"Only final conversation file saved: {conversation_filename}")
        print(f"No intermediate speaker files were saved.")
        
        return conversation_filepath, script_text
    else:
        raise gr.Error("No audio generated. Please check your inputs.")

def update_speaker_voices(script_text, *voice_assignments):
    """Update speaker voice assignments and return updated components"""
    conversation = parse_conversation_script(script_text)
    if not conversation:
        return [], {}
    
    speakers = list(set([speaker for speaker, _ in conversation]))
    
    # Create voice assignment dictionary
    speaker_voices = {}
    for i, speaker in enumerate(speakers):
        if i < len(voice_assignments):
            speaker_voices[speaker] = voice_assignments[i]
    
    return speakers, speaker_voices

# Function to generate conversation audio
def generate_conversation(speaker1_name, speaker1_voice, speaker1_text, speaker1_speed,
                         speaker2_name, speaker2_voice, speaker2_text, speaker2_speed,
                         speaker3_name, speaker3_voice, speaker3_text, speaker3_speed,
                         speaker4_name, speaker4_voice, speaker4_text, speaker4_speed,
                         speaker5_name, speaker5_voice, speaker5_text, speaker5_speed,
                         pause_duration):
    
    # Collect all speakers and their data
    speakers = [
        (speaker1_name, speaker1_voice, speaker1_text, speaker1_speed),
        (speaker2_name, speaker2_voice, speaker2_text, speaker2_speed),
        (speaker3_name, speaker3_voice, speaker3_text, speaker3_speed),
        (speaker4_name, speaker4_voice, speaker4_text, speaker4_speed),
        (speaker5_name, speaker5_voice, speaker5_text, speaker5_speed)
    ]
    
    # Filter out speakers with no text
    active_speakers = [(name, voice, text, speed) for name, voice, text, speed in speakers if text.strip()]
    
    if not active_speakers:
        raise gr.Error("Please add text for at least one speaker.")
    
    conversation_script = []
    audio_segments = []
    
    # Generate pause audio (silence)
    pause_samples = int(24000 * pause_duration)  # 24kHz sample rate
    pause_audio = torch.zeros(pause_samples)
    
    for i, (name, voice, text, speed) in enumerate(active_speakers):
        # Update conversation script
        speaker_name = name.strip() if name.strip() else f"Speaker {i+1}"
        conversation_script.append(f"{speaker_name}: {text}")
        
        # Generate audio for this speaker
        try:
            audio_path, _, _ = generate_first(text, voice, speed)
            
            # Load the generated audio
            sample_rate, audio_data = read(audio_path)
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            
            # Normalize audio
            if audio_tensor.max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.max()
            
            audio_segments.append(audio_tensor)
            
            # Add pause after each speaker (except the last one)
            if i < len(active_speakers) - 1:
                audio_segments.append(pause_audio)
                
        except Exception as e:
            raise gr.Error(f"Error generating audio for {speaker_name}: {str(e)}")
    
    # Combine all audio segments
    if audio_segments:
        combined_audio = torch.cat(audio_segments, dim=-1)
        combined_audio_numpy = combined_audio.detach().cpu().numpy()
        
        # Save the combined conversation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_filename = f"conversation_{timestamp}.wav"
        conversation_filepath = os.path.join(output_folder, conversation_filename)
        
        print(f"Saving conversation as: {conversation_filename}")
        write(conversation_filepath, 24000, combined_audio_numpy)
        actual_file_size_mb = os.path.getsize(conversation_filepath) / (1024 * 1024)
        print(f"Conversation saved successfully! Size: {actual_file_size_mb:.1f} MB")
        
        # Create conversation script text
        script_text = "\n".join(conversation_script)
        
        # Check if file is too large and provide information
        conversation_length_seconds = len(combined_audio_numpy) / 24000
        
        print(f"🎬 Conversation generation complete!")
        print(f"Total audio length: {conversation_length_seconds:.1f} seconds ({conversation_length_seconds/60:.1f} minutes)")
        print(f"Speakers processed: {len(active_speakers)}")
        
        if actual_file_size_mb > 50:
            print(f"⚠️  Large conversation file generated ({actual_file_size_mb:.1f} MB)")
            print(f"📁 File location: {conversation_filepath}")
            print(f"💡 Note: Large files may not display waveforms properly in the browser.")
        
        return conversation_filepath, script_text
    else:
        raise gr.Error("No audio generated. Please check your inputs.")

def debug_custom_voices():
    """Debug function to list custom voice files"""
    print("\n=== CUSTOM VOICES DEBUG ===")
    print(f"Custom voices folder: {custom_voices_folder}")
    print(f"Folder exists: {os.path.exists(custom_voices_folder)}")
    
    if os.path.exists(custom_voices_folder):
        all_files = os.listdir(custom_voices_folder)
        pt_files = [f for f in all_files if f.endswith('.pt')]
        print(f"All files in folder: {all_files}")
        print(f"PT files found: {pt_files}")
        
        # Check what get_custom_voices() returns
        custom_voices_dict = get_custom_voices()
        print(f"get_custom_voices() result: {custom_voices_dict}")
        
        # Check loaded voices
        custom_loaded = {k: v for k, v in loaded_voices.items() if k.startswith('custom_')}
        print(f"Loaded custom voices: {list(custom_loaded.keys())}")
    else:
        print("Custom voices folder does not exist!")
    print("=== END DEBUG ===\n")

# Function to get voice choices for dropdowns
def get_voice_choices():
    updated_choices = update_voice_choices()
    return list(updated_choices.keys())

# Function to clear conversation inputs
def clear_conversation():
    return ("", "", "", 1.0, "", "", "", 1.0, "", "", "", 1.0, "", "", "", 1.0, "", "", "", 1.0, None, "")

def clear_script_conversation():
    """Clear the script-based conversation inputs"""
    return "", None, "", []

def create_voice_assignment_interface(script_text):
    """Create dynamic voice assignment interface based on detected speakers"""
    conversation = parse_conversation_script(script_text)
    if not conversation:
        # Return updates for: voice_assignment_interface + 10 radio buttons + detected_speakers
        empty_updates = [gr.update(visible=False) for _ in range(10)]
        return [gr.update(visible=False)] + empty_updates + [[]]
    
    speakers = list(set([speaker for speaker, _ in conversation]))
    speakers.sort()  # Sort for consistent ordering
    
    # Create individual updates for each radio button
    radio_updates = []
    voice_choices = get_voice_choices()  # This includes both standard and custom voices
    
    # Debug: Print available voices
    print(f"Available voice choices: {len(voice_choices)} voices")
    custom_count = len([v for v in voice_choices if v.startswith('👤 Custom:')])
    print(f"Custom voices found: {custom_count}")
    
    for i in range(10):  # Max 10 speakers supported
        if i < len(speakers):
            radio_updates.append(gr.update(
                visible=True, 
                label=f"🎤 {speakers[i]}", 
                value=voice_choices[i % len(voice_choices)],
                choices=voice_choices
            ))
        else:
            radio_updates.append(gr.update(visible=False))
    
    # Return: voice_assignment_interface update + 10 individual radio updates + detected_speakers
    return [gr.update(visible=True)] + radio_updates + [speakers]

def generate_from_script_with_voices(script_text, pause_duration, default_speed, output_format, *voice_assignments):
    """Generate conversation from script with voice assignments"""
    conversation = parse_conversation_script(script_text)
    if not conversation:
        raise gr.Error("No conversation found in the script.")
    
    speakers = list(set([speaker for speaker, _ in conversation]))
    speakers.sort()
    
    # Create speaker-to-voice mapping
    speaker_voices = {}
    for i, speaker in enumerate(speakers):
        if i < len(voice_assignments) and voice_assignments[i]:
            speaker_voices[speaker] = voice_assignments[i]
    
    # Check if all speakers have voices assigned
    missing_voices = [speaker for speaker in speakers if speaker not in speaker_voices]
    if missing_voices:
        raise gr.Error(f"Please assign voices for: {', '.join(missing_voices)}")
    
    return generate_conversation_from_script(script_text, speaker_voices, pause_duration, default_speed, output_format)

with gr.Blocks(css="""
            /* Background animation */
            @keyframes gradientBG {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            /* Glow effects */
            @keyframes glow {
                0% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
                50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8), 0 0 30px rgba(102, 126, 234, 0.6); }
                100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
            }
            
            @keyframes textGlow {
                0% { text-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
                50% { text-shadow: 0 0 15px rgba(102, 126, 234, 0.8), 0 0 25px rgba(102, 126, 234, 0.6); }
                100% { text-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
            }
            
            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
            
            body {
                background: linear-gradient(135deg, #0f1724, #1a1f35);
                background-size: 400% 400%;
                animation: gradientBG 15s ease infinite;
                margin: 0;
                padding: 20px;
                font-family: 'Poppins', sans-serif;
                color: #f5f5f5;
                min-height: 100vh;
            }
            
            .gradio-container {
                background: rgba(20, 25, 40, 0.7);
                border-radius: 16px;
                backdrop-filter: blur(10px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                padding: 1.5rem;
                max-width: 100%;
                width: 2000px;
                margin: 0 auto;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            /* Button styling */
            button.primary {
                background: linear-gradient(45deg, #667eea, #764ba2);
                border: none;
                color: white;
                padding: 0.8rem 1.5rem;
                border-radius: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
                text-transform: uppercase;
                font-size: 0.9rem;
                letter-spacing: 0.5px;
                position: relative;
                overflow: hidden;
                margin: 0.5rem 0;
            }
            
            button.primary:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
                animation: glow 3s infinite;
            }
            
            button.primary::after {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(
                    to right,
                    rgba(255, 255, 255, 0) 0%,
                    rgba(255, 255, 255, 0.3) 50%,
                    rgba(255, 255, 255, 0) 100%
                );
                transform: rotate(30deg);
                animation: shimmer 3s infinite;
                opacity: 0;
                transition: opacity 0.3s;
            }
            
            button.primary:hover::after {
                opacity: 1;
            }
            
            /* Card styling */
            .card {
                background: rgba(30, 35, 50, 0.5);
                border-radius: 16px;
                padding: 1.8rem;
                margin: 1rem 0;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2), 0 0 15px rgba(102, 126, 234, 0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(5px);
                position: relative;
                overflow: hidden;
            }
            
            .card::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(125deg, 
                    rgba(255, 255, 255, 0) 0%, 
                    rgba(255, 255, 255, 0.05) 40%, 
                    rgba(255, 255, 255, 0) 80%);
                transform: translateX(-100%);
                transition: transform 0.7s ease;
            }
            
            .card:hover::after {
                transform: translateX(100%);
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3), 0 0 20px rgba(102, 126, 234, 0.4);
            }
            
            /* Stat box styling */
            .stat-box {
                background: rgba(30, 35, 50, 0.6);
                border-radius: 12px;
                padding: 15px 20px;
                text-align: center;
                backdrop-filter: blur(5px);
                transition: all 0.3s ease;
                border: 1px solid rgba(102, 126, 234, 0.15);
                position: relative;
                overflow: hidden;
                margin: 10px 0;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2), 0 0 10px rgba(102, 126, 234, 0.2);
            }
            
            .stat-box:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25), 0 0 15px rgba(102, 126, 234, 0.4);
                border-color: rgba(102, 126, 234, 0.3);
            }
            
            /* Voice code styling */
            .voice-code {
                font-family: monospace;
                background: linear-gradient(45deg, #4776E6, #8E54E9);
                color: white;
                padding: 5px 10px;
                border-radius: 8px;
                font-weight: bold;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2), 0 0 10px rgba(102, 126, 234, 0.3);
                display: inline-block;
                margin: 3px 5px;
                transition: all 0.3s ease;
            }
            
            .voice-code:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 10px rgba(0,0,0,0.3), 0 0 15px rgba(102, 126, 234, 0.5);
            }
            
            /* Input field styling */
            input, textarea, select {
                background: rgba(20, 25, 40, 0.6) !important;
                border: 1px solid rgba(102, 126, 234, 0.2) !important;
                border-radius: 10px !important;
                color: white !important;
                transition: all 0.3s ease !important;
            }
            
            input:focus, textarea:focus, select:focus {
                border-color: rgba(102, 126, 234, 0.6) !important;
                box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
                outline: none !important;
            }
            
            /* Tab styling */
            .tab-nav {
                background: rgba(30, 35, 50, 0.7) !important;
                border-radius: 12px !important;
                padding: 5px !important;
                margin-bottom: 20px !important;
                border: 1px solid rgba(255, 255, 255, 0.05) !important;
            }
            
            .tab-nav button {
                border-radius: 8px !important;
                margin: 5px !important;
                transition: all 0.3s ease !important;
            }
            
            .tab-nav button.selected {
                background: linear-gradient(45deg, #667eea, #764ba2) !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
            }
            
            /* Slider styling */
            input[type=range] {
                height: 6px !important;
                background: rgba(102, 126, 234, 0.3) !important;
            }
            
            input[type=range]::-webkit-slider-thumb {
                background: linear-gradient(45deg, #667eea, #764ba2) !important;
                box-shadow: 0 0 10px rgba(102, 126, 234, 0.5) !important;
            }
            
            /* Checkbox styling */
            input[type=checkbox] {
                accent-color: #667eea !important;
                width: 16px !important;
                height: 16px !important;
                margin-right: 8px !important;
            }
            
            input[type=checkbox]:checked {
                background-color: #667eea !important;
                border-color: #667eea !important;
                box-shadow: 0 0 10px rgba(102, 126, 234, 0.5) !important;
            }
            
            /* Radio button styling */
            input[type=radio] {
                accent-color: #667eea !important;
                width: 16px !important;
                height: 16px !important;
                margin-right: 8px !important;
            }
            
            input[type=radio]:checked {
                background-color: #667eea !important;
                border-color: #667eea !important;
                box-shadow: 0 0 10px rgba(102, 126, 234, 0.5) !important;
            }
            
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(30, 35, 50, 0.5);
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(45deg, #667eea, #764ba2);
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(45deg, #5a71d6, #6a43a9);
            }
            
            /* Audio player styling */
            audio {
                width: 100% !important;
                border-radius: 12px !important;
                background: rgba(30, 35, 50, 0.7) !important;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
            }
            
            /* Dropdown styling */
            select {
                background-color: rgba(30, 35, 50, 0.7) !important;
                border: 1px solid rgba(102, 126, 234, 0.2) !important;
                border-radius: 10px !important;
                color: white !important;
                padding: 10px !important;
            }
            
            /* Tooltip styling */
            [data-tooltip]:hover::before {
                background: rgba(30, 35, 50, 0.9) !important;
                border: 1px solid rgba(102, 126, 234, 0.3) !important;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3) !important;
            }
            
            /* Make the interface more responsive */
            @media (max-width: 1600px) {
                .tips-grid {
                    grid-template-columns: repeat(2, 1fr) !important;
                }
            }
            
            @media (max-width: 768px) {
                .gradio-container {
                    padding: 1rem;
                    max-width: 98%;
                }
                
                .card {
                    padding: 1.2rem;
                }
                
                h1 {
                    font-size: 1.8rem;
                }
                
                .tips-grid {
                    grid-template-columns: 1fr !important;
                }
            }
            
            /* Improve layout spacing */
            .container {
                width: 100%;
                max-width: 100%;
                padding: 0;
            }
            
            .row {
                margin: 0 -15px;
                width: calc(100% + 30px);
            }
            
            .col {
                padding: 0 15px;
            }
            
            /* Reduce card padding for better space utilization */
            .card {
                padding: 1.5rem;
            }
            
            /* Adjust the voice mixer grid to show more columns */
            @media (min-width: 1600px) {
                .voice-mixer-grid {
                    grid-template-columns: repeat(5, 1fr) !important;
                }
            }
            
            /* Fix dropdown menu display */
            .gradio-dropdown {
                position: relative !important;
                z-index: 999999 !important;
            }
            
            .gradio-dropdown > div {
                max-height: 400px !important;
                overflow-y: auto !important;
                position: relative !important;
                z-index: 999999 !important;
            }
            
            .gradio-dropdown [data-testid="dropdown-container"] {
                position: relative !important;
                z-index: 999999 !important;
            }
            
            .gradio-dropdown ul {
                max-height: 400px !important;
                overflow-y: auto !important;
                background: rgba(30, 35, 50, 0.95) !important;
                backdrop-filter: blur(10px) !important;
                border: 1px solid rgba(102, 126, 234, 0.3) !important;
                border-radius: 10px !important;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
                padding: 8px 0 !important;
                position: absolute !important;
                top: 100% !important;
                left: 0 !important;
                right: 0 !important;
                z-index: 999999 !important;
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
            }
            
            /* Additional fixes for dropdown visibility */
            .gradio-dropdown .options {
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
                position: absolute !important;
                top: 100% !important;
                left: 0 !important;
                width: 100% !important;
                z-index: 9999999 !important;
            }
            
            /* Restore missing styles */
            .gradio-dropdown li {
                padding: 8px 15px !important;
                color: white !important;
                transition: all 0.2s ease !important;
                cursor: pointer !important;
            }
            
            .gradio-dropdown li:hover {
                background: rgba(102, 126, 234, 0.2) !important;
            }
            
            /* Ensure proper stacking context */
            #generate-tab, #custom-voices-tab, #voice-mixer-tab {
                position: relative !important;
                z-index: 1 !important;
            }
            
            /* Fix for select dropdown */
            select {
                position: relative !important;
                z-index: 999999 !important;
                appearance: auto !important;
                -webkit-appearance: auto !important;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E") !important;
                background-repeat: no-repeat !important;
                background-position: right 10px center !important;
                padding-right: 30px !important;
            }
            
            /* Specific fix for voice selector dropdown */
            #voice-select {
                position: relative !important;
            }
            
            #voice-select > div {
                position: relative !important;
            }
            
            #voice-select select {
                position: relative !important;
                z-index: 999999 !important;
            }
            
            #voice-select ul {
                position: absolute !important;
                top: 100% !important;
                left: 0 !important;
                width: 100% !important;
                background: rgb(30, 35, 50) !important;
                border: 1px solid rgba(102, 126, 234, 0.3) !important;
                border-radius: 10px !important;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
                z-index: 999999 !important;
                max-height: 300px !important;
                overflow-y: auto !important;
            }
            
            #voice-select li {
                padding: 8px 12px !important;
                color: white !important;
                cursor: pointer !important;
            }
            
            #voice-select li:hover {
                background: rgba(102, 126, 234, 0.2) !important;
            }
            
            /* Input controls alignment */
            .input-box .row {
                margin-bottom: 12px;
            }
            
            .input-box .gradio-dropdown,
            .input-box .gradio-slider {
                margin: 8px 0;
            }
            
            /* Button alignment */
            .input-box button {
                margin: 8px 0;
                width: auto;
                min-width: 120px;
            }
            
            /* Consistent heights for dropdowns and sliders */
            .gradio-dropdown > div,
            .gradio-slider > div {
                min-height: 40px;
            }
            
            /* Fix spacing between elements */
            .gradio-row {
                gap: 16px;
            }
            
            /* Ensure labels are aligned */
            .gradio-dropdown label,
            .gradio-slider label {
                margin-bottom: 8px;
                display: block;
            }
            
            /* Container padding */
            .input-box {
                padding: 16px;
            }
            
            /* Refresh button alignment */
            .refresh-btn-container {
                display: flex;
                justify-content: flex-start;
                margin-top: 8px;
            }
            
            #voice-select {
                position: relative !important;
            }
            
            #voice-select > div {
                position: relative !important;
            }
            
            #voice-select select {
                position: relative !important;
                z-index: 999999 !important;
            }
            
            #voice-select ul {
                position: absolute !important;
                top: 100% !important;
                left: 0 !important;
                width: 100% !important;
                background: rgb(30, 35, 50) !important;
                border: 1px solid rgba(102, 126, 234, 0.3) !important;
                border-radius: 10px !important;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
                z-index: 999999 !important;
                max-height: 300px !important;
                overflow-y: auto !important;
            }
            
            #voice-select li {
                padding: 8px 12px !important;
                color: white !important;
                cursor: pointer !important;
            }
            
            #voice-select li:hover {
                background: rgba(102, 126, 234, 0.2) !important;
            }
            
            /* Voice mixer grid styling */
            .voice-mixer-grid {
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 15px;
                margin-bottom: 15px;
            }
            
            .voice-mixer-card {
                background: rgba(30, 35, 50, 0.6);
                border-radius: 10px;
                padding: 12px;
                border: 1px solid rgba(102, 126, 234, 0.15);
                transition: all 0.3s ease;
                height: 100%;
                display: flex;
                flex-direction: column;
            }
            
            .voice-mixer-card:hover {
                border-color: rgba(102, 126, 234, 0.4);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2), 0 0 10px rgba(102, 126, 234, 0.3);
                transform: translateY(-2px);
            }
            
            .voice-mixer-card.selected {
                border-color: rgba(102, 126, 234, 0.8);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3), 0 0 15px rgba(102, 126, 234, 0.5);
                background: rgba(40, 45, 65, 0.7);
            }
            
            .voice-mixer-card label {
                margin-bottom: 8px;
                font-size: 0.9rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .voice-mixer-card .gradio-slider {
                margin-top: auto;
            }
            
            /* Ensure consistent heights for voice mixer elements */
            #voice-mixer-tab .gradio-checkbox {
                min-height: 30px;
            }
            
            #voice-mixer-tab .gradio-slider {
                min-height: 50px;
            }
            
            /* Improve spacing in voice mixer */
            #voice-mixer-tab .gradio-row {
                margin-bottom: 10px;
            }
        """) as app:
    # Add custom CSS for the new design
    app.load(js="""
    function() {
        const style = document.createElement('style');
        style.textContent = `
            /* Background animation */
            @keyframes gradientBG {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            body {
                background: linear-gradient(135deg, #0f1724, #1a1f35);
                background-size: 400% 400%;
                animation: gradientBG 15s ease infinite;
                margin: 0;
                padding: 20px;
                font-family: 'Poppins', sans-serif;
                color: #f5f5f5;
                min-height: 100vh;
            }
            
            .gradio-container {
                background: rgba(20, 25, 40, 0.7);
                border-radius: 16px;
                backdrop-filter: blur(10px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                padding: 1.5rem;
                max-width: 100%;
                width: 2000px;
                margin: 0 auto;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            /* Fix for Tips & Tricks page */
            ul, ol {
                color: #f5f5f5 !important;
                margin-left: 0;
                padding-left: 20px;
            }
            
            ul li, ol li {
                margin-bottom: 8px;
                color: #f5f5f5 !important;
                display: list-item !important;
            }
            
            ul {
                list-style-type: circle !important;
            }
            
            ol {
                list-style-type: decimal !important;
            }
            
            /* Fix for code display in Tips & Tricks */
            pre, code {
                background: transparent !important;
                border: none !important;
                color: #f5f5f5 !important;
                font-family: monospace !important;
                white-space: pre-wrap !important;
                display: inline !important;
            }
            
            /* Ensure proper rendering of HTML content */
            .prose {
                color: #f5f5f5 !important;
            }
            
            .prose h1, .prose h2, .prose h3, .prose h4, .prose h5, .prose h6 {
                color: #f5f5f5 !important;
            }
            
            .prose p, .prose li {
                color: #f5f5f5 !important;
            }
            
            /* Card styling */
            .card {
                background: rgba(30, 35, 50, 0.5);
                border-radius: 16px;
                padding: 1.8rem;
                margin: 1rem 0;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2), 0 0 15px rgba(102, 126, 234, 0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(5px);
                position: relative;
                overflow: hidden;
            }
            
            .card::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(125deg, 
                    rgba(255, 255, 255, 0) 0%, 
                    rgba(255, 255, 255, 0.05) 40%, 
                    rgba(255, 255, 255, 0) 80%);
                transform: translateX(-100%);
                transition: transform 0.7s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
            }
            
            .card:hover::after {
                transform: translateX(100%);
            }
            
            /* Glow effects */
            @keyframes glow {
                0% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
                50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8), 0 0 30px rgba(102, 126, 234, 0.6); }
                100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
            }
            
            @keyframes textGlow {
                0% { text-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
                50% { text-shadow: 0 0 15px rgba(102, 126, 234, 0.8), 0 0 25px rgba(102, 126, 234, 0.6); }
                100% { text-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
            }
            
            /* Button styling */
            button {
                background: linear-gradient(45deg, #667eea, #764ba2);
                border: none;
                color: white;
                padding: 0.8rem 1.5rem;
                border-radius: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
                text-transform: uppercase;
                font-size: 0.9rem;
                letter-spacing: 0.5px;
                position: relative;
                overflow: hidden;
                margin: 0.5rem 0;
            }
            
            button::after {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(transparent, rgba(255, 255, 255, 0.2), transparent);
                transform: rotate(30deg);
                transition: 0.6s;
                opacity: 0;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
            }
            
            button:hover::after {
                left: 100%;
                opacity: 0.3;
            }
            
            #generate-btn {
                animation: glow 2.5s infinite;
                background: linear-gradient(45deg, #4776E6, #8E54E9);
                font-size: 1rem;
                padding: 1rem 1.8rem;
                margin-top: 1rem;
                width: 100%;
            }
            
            /* Input styling */
            input, textarea, select {
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 12px;
                padding: 0.8rem;
                background: rgba(30, 35, 50, 0.5);
                backdrop-filter: blur(5px);
                transition: all 0.3s ease;
                color: #f5f5f5;
                font-weight: 400;
                margin: 0.5rem 0;
            }
            
            input:focus, textarea:focus, select:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
                outline: none;
            }
            
            /* Typography */
            h1 {
                font-size: 2.2rem;
                font-weight: 700;
                color: #fff;
                margin-bottom: 1rem;
            }
            
            h2 {
                font-size: 1.8rem;
                font-weight: 600;
                color: #fff;
                margin-bottom: 1.2rem;
            }
            
            h3 {
                font-size: 1.3rem;
                font-weight: 600;
                color: #fff;
                margin: 1rem 0 0.8rem 0;
            }
            
            p {
                color: rgba(255, 255, 255, 0.8);
                line-height: 1.6;
                font-size: 1rem;
                margin-bottom: 1rem;
            }
            
            /* Tab styling */
            .tab {
                background: rgba(30, 35, 50, 0.5);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
            
            /* Selector styling */
            select, .gradio-dropdown {
                position: relative;
                border-radius: 12px;
                background: rgba(30, 35, 50, 0.5);
                border: 1px solid rgba(102, 126, 234, 0.3);
                padding: 10px;
                color: #f5f5f5;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }
            
            /* Slider styling */
            input[type=range] {
                appearance: none;
                height: 6px;
                background: linear-gradient(90deg, #4776E6, #8E54E9);
                border-radius: 6px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
            }
            
            input[type=range]::-webkit-slider-thumb {
                appearance: none;
                width: 20px;
                height: 20px;
                background: #fff;
                border-radius: 50%;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                transition: all 0.2s;
            }
            
            input[type=range]::-webkit-slider-thumb:hover {
                transform: scale(1.1);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            }
            
            /* Label styling */
            label {
                font-weight: 500;
                color: rgba(255, 255, 255, 0.9);
                margin-bottom: 6px;
                display: block;
                font-size: 1rem;
            }
            
            /* Audio player styling */
            audio {
                width: 100%;
                border-radius: 12px;
                background: rgba(30, 35, 50, 0.5);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            
            /* Animation for elements */
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            /* Apply animations to elements */
            .gradio-container > * {
                animation: fadeInUp 0.5s ease-out;
                animation-fill-mode: both;
            }
            
            .gradio-container > *:nth-child(1) { animation-delay: 0.05s; }
            .gradio-container > *:nth-child(2) { animation-delay: 0.1s; }
            .gradio-container > *:nth-child(3) { animation-delay: 0.15s; }
            .gradio-container > *:nth-child(4) { animation-delay: 0.2s; }
            
            /* Table styling */
            table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0 6px;
                margin: 15px 0;
            }
            
            th {
                text-align: left;
                padding: 10px 12px;
                background: rgba(102, 126, 234, 0.15);
                color: #fff;
                font-weight: 500;
                border-radius: 8px;
            }
            
            td {
                padding: 12px;
                background: rgba(30, 35, 50, 0.5);
                color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                transition: all 0.3s;
            }
            
            tr:hover td {
                background: rgba(102, 126, 234, 0.15);
                transform: scale(1.01);
            }
            
            /* Voice code styling */
            .voice-code {
                font-family: monospace;
                background: linear-gradient(45deg, #4776E6, #8E54E9);
                color: white;
                padding: 5px 10px;
                border-radius: 8px;
                font-weight: bold;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2), 0 0 10px rgba(102, 126, 234, 0.3);
                display: inline-block;
                margin: 3px 5px;
                transition: all 0.3s ease;
            }
            
            .voice-code:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 10px rgba(0,0,0,0.3), 0 0 15px rgba(102, 126, 234, 0.5);
            }
            
            /* Loader animation */
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .generating::after {
                content: "";
                display: inline-block;
                width: 16px;
                height: 16px;
                margin-left: 8px;
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 1s ease-in-out infinite;
            }
            
            /* Wave animation for audio */
            @keyframes wave {
                0% { height: 4px; }
                50% { height: 16px; }
                100% { height: 4px; }
            }
            
            .audio-visualizer {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 25px;
                margin: 8px 0;
            }
            
            .audio-visualizer span {
                display: inline-block;
                width: 4px;
                height: 4px;
                margin: 0 2px;
                background: linear-gradient(45deg, #4776E6, #8E54E9);
                border-radius: 4px;
                animation: wave 1.5s ease infinite;
            }
            
            .audio-visualizer span:nth-child(2) { animation-delay: 0.2s; }
            .audio-visualizer span:nth-child(3) { animation-delay: 0.4s; }
            .audio-visualizer span:nth-child(4) { animation-delay: 0.6s; }
            .audio-visualizer span:nth-child(5) { animation-delay: 0.8s; }
            
            /* Stats box styling */
            .stat-box {
                background: rgba(30, 35, 50, 0.6);
                border-radius: 12px;
                padding: 15px 20px;
                text-align: center;
                backdrop-filter: blur(5px);
                transition: all 0.3s ease;
                border: 1px solid rgba(102, 126, 234, 0.15);
                position: relative;
                overflow: hidden;
                margin: 10px 0;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2), 0 0 10px rgba(102, 126, 234, 0.2);
            }
            
            .stat-box:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25), 0 0 15px rgba(102, 126, 234, 0.4);
                border-color: rgba(102, 126, 234, 0.3);
            }
            
            .stat-box::after {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(transparent, rgba(102, 126, 234, 0.1), transparent);
                transform: rotate(30deg);
                transition: 0.6s;
                opacity: 0;
            }
            
            .stat-box:hover::after {
                left: 100%;
                opacity: 0.5;
            }
            
            .stat-number {
                font-size: 1.5rem;
                font-weight: bold;
                color: #fff;
                margin: 0;
                background: linear-gradient(90deg, #4776E6, #8E54E9);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .stat-label {
                font-size: 0.85rem;
                color: rgba(255, 255, 255, 0.7);
                margin: 3px 0 0 0;
            }
            
            /* Tab button styling */
            .tab-nav button {
                background: transparent;
                box-shadow: none;
                border-bottom: 2px solid transparent;
                border-radius: 0;
                padding: 0.6rem 1rem;
                margin: 0 0.3rem;
                transition: all 0.3s ease;
            }
            
            .tab-nav button:hover, .tab-nav button.selected {
                border-color: #667eea;
                background: rgba(102, 126, 234, 0.1);
                transform: translateY(0);
            }
            
            .tab-nav button.selected {
                position: relative;
            }
            
            .tab-nav button.selected::after {
                content: '';
                position: absolute;
                bottom: -2px;
                left: 0;
                width: 100%;
                height: 2px;
                background: linear-gradient(90deg, #4776E6, #8E54E9);
                animation: glow 2.5s infinite;
            }
            
            /* Make accordion look better */
            .accordion {
                background: transparent;
                border: 1px solid rgba(102, 126, 234, 0.15);
                border-radius: 12px;
                overflow: hidden;
                margin: 1rem 0;
            }
            
            .accordion button {
                background: rgba(30, 35, 50, 0.6);
                border-radius: 0;
                width: 100%;
                text-align: left;
                padding: 0.8rem 1rem;
                box-shadow: none;
            }
            
            /* Fix for dropdown visibility */
            #voice-select .wrap.svelte-1p9xokt {
                position: relative !important;
                z-index: 9999999 !important;
            }
            
            #voice-select .wrap.svelte-1p9xokt .options.svelte-1p9xokt {
                position: absolute !important;
                top: 100% !important;
                left: 0 !important;
                width: 100% !important;
                z-index: 9999999 !important;
                background: rgba(30, 35, 50, 0.95) !important;
                border: 1px solid rgba(102, 126, 234, 0.3) !important;
                border-radius: 10px !important;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
                max-height: 300px !important;
                overflow-y: auto !important;
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
            }
        `;
        document.head.appendChild(style);
        
        // Add audio visualizer elements when audio is playing
        setTimeout(() => {
            const audioElements = document.querySelectorAll('audio');
            audioElements.forEach(audio => {
                const container = document.createElement('div');
                container.className = 'audio-visualizer';
                container.innerHTML = '<span></span><span></span><span></span><span></span><span></span>';
                
                const parent = audio.parentNode;
                parent.insertBefore(container, audio);
                
                audio.addEventListener('play', () => {
                    container.style.display = 'flex';
                });
                
                audio.addEventListener('pause', () => {
                    container.style.display = 'none';
                });
                
                // Initially hide
                container.style.display = 'none';
            });
        }, 1000);
        
        // Add animation to the generate button
        setTimeout(() => {
            const generateBtn = document.querySelector('#generate-btn');
            if (generateBtn) {
                generateBtn.addEventListener('click', function() {
                    this.classList.add('generating');
                    setTimeout(() => {
                        this.classList.remove('generating');
                    }, 5000);
                });
            }
        }, 1000);
        
        // Fix dropdown visibility
        setTimeout(() => {
            // Fix for dropdown visibility
            const voiceSelect = document.querySelector('#voice-select');
            if (voiceSelect) {
                // Make sure the dropdown container has proper z-index
                voiceSelect.style.position = 'relative';
                voiceSelect.style.zIndex = '9999999';
                
                // Find all dropdown elements and ensure they're visible
                const dropdownElements = voiceSelect.querySelectorAll('.wrap');
                dropdownElements.forEach(el => {
                    el.style.position = 'relative';
                    el.style.zIndex = '9999999';
                    
                    // Find dropdown options
                    const options = el.querySelector('.options');
                    if (options) {
                        options.style.position = 'absolute';
                        options.style.top = '100%';
                        options.style.left = '0';
                        options.style.width = '100%';
                        options.style.zIndex = '9999999';
                        options.style.background = 'rgba(30, 35, 50, 0.95)';
                        options.style.border = '1px solid rgba(102, 126, 234, 0.3)';
                        options.style.borderRadius = '10px';
                        options.style.boxShadow = '0 10px 25px rgba(0, 0, 0, 0.3)';
                        options.style.maxHeight = '300px';
                        options.style.overflowY = 'auto';
                    }
                });
            }
        }, 1000);
        
        // Add highlighting for voice mixer cards when checkbox is checked
        setTimeout(() => {
            const voiceMixerTab = document.querySelector('#voice-mixer-tab');
            if (voiceMixerTab) {
                const checkboxes = voiceMixerTab.querySelectorAll('input[type="checkbox"]');
                checkboxes.forEach(checkbox => {
                    // Get the parent card element
                    const card = checkbox.closest('.voice-mixer-card');
                    if (card) {
                        // Set initial state
                        if (checkbox.checked) {
                            card.classList.add('selected');
                        }
                        
                        // Add event listener for changes
                        checkbox.addEventListener('change', function() {
                            if (this.checked) {
                                card.classList.add('selected');
                            } else {
                                card.classList.remove('selected');
                            }
                        });
                    }
                });
            }
        }, 1000);
    }
    """)

    with gr.Row(variant="panel", elem_id="header-section"):
        gr.Markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.8rem; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 300px; text-align: center;">
                    <h1 style="font-size: 2.2rem; margin: 0; text-shadow: 0 0 15px rgba(102, 126, 234, 0.8);">🎙️ Kokoro TTS Local</h1>
                    <p style="font-size: 1rem; margin: 0.5rem 0 0 0; opacity: 0.9;">
                        Transform your text into natural-sounding speech with our advanced AI voice synthesis.
                    </p>
                </div>
                <div style="display: flex; justify-content: center; gap: 15px; margin: 0; flex-wrap: wrap;">
                    <div class="stat-box" style="flex: 0 0 auto; width: 120px; padding: 10px;">
                        <p class="stat-number" style="font-size: 1.3rem; margin: 0; text-shadow: 0 0 10px rgba(102, 126, 234, 0.7);">30+</p>
                        <p class="stat-label" style="font-size: 0.8rem; margin: 0;">Unique Voices</p>
                    </div>
                    <div class="stat-box" style="flex: 0 0 auto; width: 120px; padding: 10px;">
                        <p class="stat-number" style="font-size: 1.3rem; margin: 0; text-shadow: 0 0 10px rgba(102, 126, 234, 0.7);">3</p>
                        <p class="stat-label" style="font-size: 0.8rem; margin: 0;">Languages</p>
                    </div>
                    <div class="stat-box" style="flex: 0 0 auto; width: 120px; padding: 10px;">
                        <p class="stat-number" style="font-size: 1.3rem; margin: 0; text-shadow: 0 0 10px rgba(102, 126, 234, 0.7);">100%</p>
                        <p class="stat-label" style="font-size: 0.8rem; margin: 0;">Local Processing</p>
                    </div>
                </div>
            </div>
            """
        )
    
    with gr.Tabs(elem_id="main-tabs"):
        with gr.TabItem("🎤 Generate Speech", elem_id="generate-tab"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    with gr.Column(elem_id="input-box", elem_classes=["card"]):
                        text = gr.Textbox(
                            label='✍️ Enter Your Text', 
                            placeholder="Type something here to convert to speech...", 
                            lines=8,
                            elem_id="text-input"
                        )
                        
                        generate_btn = gr.Button('🔊 Generate Speech', variant='primary', elem_id="generate-btn")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                voice = gr.Radio(
                                    choices=list(CHOICES.keys()),
                                    value=list(CHOICES.keys())[0],
                                    label='👤 Select Voice',
                                    info='Choose a voice for the output',
                                    scale=1,
                                    elem_id="voice-select",
                                    container=True,
                                    interactive=True
                                )
                            with gr.Column(scale=2):
                                with gr.Row():
                                    speed = gr.Slider(
                                        minimum=0.5, 
                                        maximum=4, 
                                        value=1, 
                                        step=0.1, 
                                        label='⚡ Speech Speed', 
                                        info='Adjust speed (0.5 to 4x)',
                                        elem_id="speed-control"
                                    )
                                with gr.Row():
                                    output_format = gr.Radio(
                                        choices=['WAV', 'MP3'],
                                        value='WAV',
                                        label='🎵 Output Format',
                                        info='Choose audio file format'
                                    )
                        
                        with gr.Row():
                            refresh_btn = gr.Button('🔄 Refresh Voices To Show Custom Voices', size='sm')
                    
                with gr.Column(scale=2):
                    with gr.Column(elem_id="output-box", elem_classes=["card"]):
                        gr.Markdown("<h3 style='text-align: center; margin-top: 0; font-size: 1.2rem;'>🎧 Generated Audio</h3>")
                        
                        out_audio = gr.Audio(
                            label=None, 
                            interactive=False, 
                            streaming=False, 
                            autoplay=False,
                            elem_id="audio-output"
                        )
                        
                        large_file_info = gr.Markdown(
                            visible=False,
                            value="""
                            <div style="background: rgba(255, 193, 7, 0.1); border: 1px solid rgba(255, 193, 7, 0.3); border-radius: 8px; padding: 12px; margin: 10px 0;">
                                <h4 style="margin-top: 0; color: #ffc107;">📁 Large File Generated</h4>
                                <p style="margin-bottom: 0;">This file is quite large and may not display properly in the browser waveform. 
                                You can find the complete audio file in the <strong>outputs</strong> folder for the best playback experience.</p>
                            </div>
                            """,
                            elem_id="large-file-warning"
                        )
                        
                        with gr.Accordion("Advanced Details", open=False):
                            out_ps = gr.Textbox(
                                interactive=False, 
                                label="🔤 Phoneme Sequence", 
                                info='The phoneme sequence corresponding to the input text',
                                elem_id="phoneme-output"
                            )
            
            # Tips & Tricks section moved from its own tab to the bottom of Generate Speech tab
            with gr.Column(elem_classes=["card"]):
                gr.Markdown("<h2 style='text-align: center; margin-bottom: 20px;'>💡 Tips & Tricks</h2>")
                gr.HTML(
                    """
                    <div style="color: white; padding: 10px;">
                        <div class="tips-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 20px;">
                            <div>
                                <h3>✨ Improve Speech Quality</h3>
                                <ul style="list-style-type: circle; padding-left: 25px;">
                                    <li>Add proper punctuation for natural pauses</li>
                                    <li>Use complete sentences for better flow</li>
                                    <li>Experiment with different speeds (0.9-1.2 often sounds most natural)</li>
                                    <li>Match voice to content type for best results</li>
                                </ul>
                            </div>
                            
                            <div>
                                <h3>🔤 Special Content Tips</h3>
                                <ul style="list-style-type: circle; padding-left: 25px;">
                                    <li>For important numbers, write them as words</li>
                                    <li>Use periods between letters in acronyms (U.S.A.)</li>
                                    <li>For technical terms, test different pronunciations</li>
                                    <li>Add hyphens to complex compound words</li>
                                </ul>
                            </div>
                        
                            <div>
                                <h3>⚡ Performance Boosters</h3>
                                <ul style="list-style-type: circle; padding-left: 25px;">
                                    <li>Keep text under 1000 characters per generation for best results</li>
                                    <li>Break longer texts into smaller, logical chunks</li>
                                    <li>Processing is faster with GPU acceleration</li>
                                    <li>Custom voices may require more processing power</li>
                                </ul>
                            </div>
                            
                            <div>
                                <h3>🚀 Quick Start Guide</h3>
                                <ol style="padding-left: 25px;">
                                    <li>Type or paste your text in the input box</li>
                                    <li>Select your preferred voice from the dropdown</li>
                                    <li>Adjust the speech speed if desired</li>
                                    <li>Click the "Generate Speech" button</li>
                                    <li>Listen to your generated audio and download if desired</li>
                                </ol>
                            </div>
                        </div>
                    </div>
                    """
                )

        with gr.TabItem("👤 Custom Voices", elem_id="custom-voices-tab"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown(
                            """
                            <h2 style="text-align: center; margin-top: 0; font-size: 1.5rem;">🎤 Upload Your Voice</h2>
                            
                            <p style="text-align: center;">
                                Add your own custom voices to use with Kokoro TTS. The system works with PyTorch (.pt) 
                                voice files compatible with the Kokoro model.
                            </p>
                            
                            <div class="stat-box" style="margin: 15px 0; padding: 15px 20px;">
                                <h3 style="margin-top: 0; font-size: 1.2rem;">📋 Steps to Add a Voice:</h3>
                                <ol style="padding-left: 20px; margin-bottom: 0; text-align: left;">
                                    <li>Prepare your .pt voice file</li>
                                    <li>Enter a unique name below</li>
                                    <li>Upload your .pt file</li>
                                    <li>Click "Upload Voice"</li>
                                    <li>Go to Generate Speech and select your custom voice</li>
                                </ol>
                            </div>
                            
                            <div class="stat-box" style="margin: 15px 0; padding: 15px 20px;">
                                <h3 style="margin-top: 0; font-size: 1.2rem;">💡 Important Notes:</h3>
                                <ul style="padding-left: 20px; margin-bottom: 0; text-align: left;">
                                    <li>Each voice must be a .pt file compatible with Kokoro</li>
                                    <li>Custom voices use the American English pipeline by default</li>
                                    <li>For best results, use high-quality voice reference files</li>
                                    <li>Your custom voices appear with a 👤 prefix in the voice selector</li>
                                </ul>
                            </div>
                            """
                        )
                        
                        custom_voice_name = gr.Textbox(
                            label='👤 Custom Voice Name', 
                            placeholder="Enter a name for your custom voice",
                            info="Use only letters, numbers, and underscores"
                        )
                        
                        custom_voice_files = gr.File(
                            label="📁 Upload Voice File", 
                            file_count="single",
                            file_types=[".pt"]
                        )
                        
                        upload_btn = gr.Button('📤 Upload Voice', variant='primary')
                        upload_status = gr.Textbox(label="📊 Upload Status", interactive=False)
                        
                        gr.Markdown(
                            """
                            <h2 style="text-align: center; margin-top: 20px; font-size: 1.5rem;">📋 Your Custom Voices</h2>
                            
                            <p style="text-align: center;">
                                Below is a list of your uploaded custom voices. Use the refresh button to update the list
                                after adding new voices.
                            </p>
                            """
                        )
                        
                        custom_voice_list = gr.Dataframe(
                            headers=["Voice Name", "Status"],
                            datatype=["str", "str"],
                            row_count=(5, "fixed"),
                            col_count=(2, "fixed"),
                            interactive=False,
                            value=get_custom_voice_list()
                        )
                        
                        refresh_custom_btn = gr.Button('🔄 Refresh List')

        with gr.TabItem("🎚️ Voice Mixer", elem_id="voice-mixer-tab"):
            with gr.Column(elem_classes=["card"]):
                gr.Markdown(
                    """
                    <h2 style="text-align: center; margin-top: 0; font-size: 1.5rem;">🎚️ Voice Mixer</h2>
                    
                    <p style="text-align: center;">
                        Create unique voices by mixing existing ones. Select voices and adjust their weights using the sliders below.
                    </p>
                    """
                )
                
                # Create a grid of voice sliders
                voice_checkboxes = []
                voice_sliders = []
                
                with gr.Row():
                    with gr.Column():
                        # Create rows of 5 voice sliders each
                        voice_keys = list(CHOICES.keys())
                        
                        # Sort the voices by their display names for better organization
                        voice_keys.sort()
                        
                        # Group voices by language/region prefix for better organization
                        us_female_voices = [k for k in voice_keys if k.startswith('🇺🇸 🚺')]
                        us_male_voices = [k for k in voice_keys if k.startswith('🇺🇸 🚹')]
                        gb_female_voices = [k for k in voice_keys if k.startswith('🇬🇧 🚺')]
                        gb_male_voices = [k for k in voice_keys if k.startswith('🇬🇧 🚹')]
                        other_voices = [k for k in voice_keys if not (k.startswith('🇺🇸') or k.startswith('🇬🇧'))]
                        
                        # Combine in a logical order
                        organized_voices = us_female_voices + us_male_voices + gb_female_voices + gb_male_voices + other_voices
                        
                        # Create rows with equal number of items for better alignment
                        items_per_row = 5
                        rows = [organized_voices[i:i+items_per_row] for i in range(0, len(organized_voices), items_per_row)]
                        
                        # Add empty placeholders to the last row if needed to maintain grid alignment
                        if len(rows) > 0 and len(rows[-1]) < items_per_row:
                            rows[-1].extend([None] * (items_per_row - len(rows[-1])))
                        
                        for row in rows:
                            with gr.Row(equal_height=True, elem_classes=["voice-mixer-grid"]):
                                for voice_name in row:
                                    with gr.Column(scale=1, min_width=150, elem_classes=["voice-mixer-card"]):
                                        if voice_name is not None:
                                            voice_id = CHOICES[voice_name]
                                            checkbox = gr.Checkbox(label=voice_name, value=False)
                                            slider = gr.Slider(
                                                minimum=0, 
                                                maximum=1, 
                                                value=0, 
                                                step=0.01, 
                                                label="Weight",
                                                interactive=True
                                            )
                                            voice_checkboxes.append(checkbox)
                                            voice_sliders.append(slider)
                                        else:
                                            # Empty placeholder to maintain grid alignment
                                            gr.Markdown("&nbsp;")
                                            gr.Markdown("&nbsp;")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        voice_formula = gr.Textbox(
                            label="🔠 Voice Formula",
                            placeholder="Formula will be generated from sliders",
                            info="This formula will be used to create the mixed voice",
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        mixed_voice_name = gr.Textbox(
                            label="🏷️ Mixed Voice Name",
                            placeholder="Enter a name for your mixed voice (optional)",
                            info="Leave blank for auto-generated name"
                        )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        voice_text = gr.Textbox(
                            label="Enter Text",
                            placeholder="Type your text here to preview the custom voice...",
                            lines=3
                        )
                    with gr.Column(scale=1):
                        mix_btn = gr.Button('🔄 Create Mixed Voice', variant='primary', size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        mix_status = gr.Textbox(label="📊 Mixing Status", interactive=False)
                    with gr.Column(scale=1):
                        mix_audio = gr.Audio(label="Preview", interactive=False)
                
                gr.Markdown(
                    """
                    <div class="stat-box" style="margin: 15px 0; padding: 15px 20px;">
                        <h3 style="margin-top: 0; font-size: 1.2rem;">💡 Tips:</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                            <ul style="padding-left: 20px; margin-bottom: 0; text-align: left;">
                                <li>Check the boxes for voices you want to include</li>
                                <li>Adjust the sliders to set the weight of each voice</li>
                                <li>Give your mixed voice a descriptive name</li>
                            </ul>
                            <ul style="padding-left: 20px; margin-bottom: 0; text-align: left;">
                                <li>Mix similar voice types for best results</li>
                                <li>The mixed voice will appear in your custom voices</li>
                                <li>Try different weight combinations for unique results</li>
                            </ul>
                        </div>
                    </div>
                    """
                )

        with gr.TabItem("📁 Batch Convert", elem_id="batch-convert-tab"):
            with gr.Column(elem_classes=["card"]):
                gr.Markdown(
                    """
                    <h2 style='text-align: center; margin-bottom: 1rem; background: linear-gradient(90deg, #4776E6, #8E54E9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold;'>
                        📁 Batch Text-to-Speech Conversion
                    </h2>
                    <p style='text-align: center; margin-bottom: 1.5rem; color: #666; font-size: 1.1rem;'>
                        Convert multiple text files to audio at once with the same voice settings
                    </p>
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["card"]):
                        gr.Markdown("<h3 style='text-align: center; margin-top: 0;'>📂 Upload Text Files</h3>")
                        
                        batch_files = gr.File(
                            label="Select Text Files (.txt)",
                            file_count="multiple",
                            file_types=[".txt"],
                            height=200
                        )
                        
                        gr.Markdown(
                            """
                            **Supported formats:** .txt files with UTF-8 encoding  
                            **Tip:** You can select multiple files at once using
                            """
                        )
                
                with gr.Row():
                    with gr.Column(scale=2, elem_classes=["card"]):
                        gr.Markdown("<h3 style='text-align: center; margin-top: 0;'>🎤 Voice Assignment for Each File</h3>")
                        
                        # Dynamic file-voice assignment interface
                        file_voice_assignments = gr.Column(visible=False)
                        
                        # Create voice assignment radios for each file (up to 20 files)
                        file_voice_radios = []
                        for i in range(20):
                            voice_radio = gr.Radio(
                                choices=list(update_voice_choices().keys()),
                                value=list(update_voice_choices().keys())[0],
                                label=f"File {i+1}",
                                visible=False,
                                interactive=True
                            )
                            file_voice_radios.append(voice_radio)
                    
                    with gr.Column(scale=1, elem_classes=["card"]):
                        gr.Markdown("<h3 style='text-align: center; margin-top: 0;'>⚙️ Global Settings</h3>")
                        
                        batch_speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speed (applies to all files)",
                            interactive=True
                        )
                        
                        batch_output_format = gr.Radio(
                            choices=['WAV', 'MP3'],
                            value='WAV',
                            label="Output Format (applies to all files)",
                            interactive=True
                        )
                        
                        gr.Markdown("---")
                        
                        # Quick voice assignment buttons
                        gr.Markdown("**Quick Assign:**")
                        
                        with gr.Row():
                            assign_same_voice_btn = gr.Button(
                                "🔄 Use Same Voice for All",
                                size="sm",
                                variant="secondary"
                            )
                        
                        quick_voice_select = gr.Radio(
                            choices=list(update_voice_choices().keys())[:28],  # Show top 5 voices for quick selection
                            value=list(update_voice_choices().keys())[0],
                            label="Quick Voice Selection",
                            interactive=True
                        )
                        
                        batch_convert_btn = gr.Button(
                            "🚀 Start Batch Conversion",
                            variant="primary",
                            size="lg"
                        )
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["card"]):
                        gr.Markdown("<h3 style='text-align: center; margin-top: 0;'>📊 Conversion Results</h3>")
                        batch_results = gr.Textbox(
                            label=None,
                            placeholder="Conversion results will appear here...",
                            lines=10,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1, elem_classes=["card"]):
                        gr.Markdown("<h3 style='text-align: center; margin-top: 0;'>🎧 Generated Audio Files</h3>")
                        batch_audio_files = gr.Column(visible=False)
                        
                        # Create multiple audio players for batch results
                        batch_audio_players = []
                        for i in range(20):  # Support up to 20 files in batch
                            audio_player = gr.Audio(
                                label=f"File {i+1}",
                                interactive=False,
                                visible=False,
                                elem_id=f"batch-audio-{i}"
                            )
                            batch_audio_players.append(audio_player)
                
                # Tips section for batch conversion
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["card"]):
                        gr.Markdown("### 💡 Batch Tips:")
                        gr.Markdown("""
• **Individual Voices:** Each file can have its own voice  
• **Quick Assignment:** Use "Same Voice for All" for consistency  
• **Output Naming:** Files are named `[original_name]_[timestamp].[format]`  
• **Auto-Skip:** Empty files are automatically skipped  
• **Processing:** Large files may take longer to process  
                        """)
                    
                    with gr.Column(scale=1, elem_classes=["card"]):
                        gr.Markdown("### 📋 How to Use:")
                        gr.Markdown("""
• **Step 1:** Upload your .txt files  
• **Step 2:** Assign voices to each file individually  
• **Step 3:** Set global speed and output format  
• **Step 4:** Click "Start Batch Conversion"  
• **Step 5:** Play generated files directly in the interface  
                        """)
                    
                    with gr.Column(scale=1, elem_classes=["card"]):
                        gr.Markdown("### 📋 File Requirements:")
                        gr.Markdown("""
• **Format:** Plain text files (.txt)  
• **Encoding:** UTF-8 recommended  
• **Size:** Up to 5000 characters per file  
• **Content:** Any text content  
• **Names:** Avoid special characters in filenames  
                        """)

        with gr.TabItem("💬 Conversation Mode", elem_id="conversation-tab"):
            with gr.Column(elem_classes=["card"]):
                gr.Markdown(
                    """
                    <h2 style="text-align: center; margin-top: 0; font-size: 1.5rem;">💬 Script-Based Conversation</h2>
                    
                    <p style="text-align: center;">
                        Simply paste your conversation script and assign voices to each speaker. The system will automatically 
                        detect speakers and generate the conversation with natural pauses between speakers.
                    </p>
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Script input
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("<h3 style='margin-top: 0;'>📝 Conversation Script</h3>")
                            conversation_script_input = gr.Textbox(
                                label=None,
                                placeholder="""Speaker 1: Hello, how are you today?
Speaker 2: I'm doing great! How about you?
Speaker 1: Pretty good, thanks for asking.
Speaker 2: That's wonderful to hear!""",
                                lines=12,
                                info="Paste your conversation script here. Use format: 'Speaker Name: dialogue'"
                            )
                            
                            # Settings
                            with gr.Row():
                                with gr.Column(scale=1):
                                    script_pause_duration = gr.Slider(
                                        minimum=-0.5, 
                                        maximum=3, 
                                        value=0.3, 
                                        step=0.1, 
                                        label='⏸️ Pause Between Speakers (seconds)',
                                        info='Use negative values to reduce gaps/overlap audio'
                                    )
                                with gr.Column(scale=1):
                                    script_speed = gr.Slider(
                                        minimum=0.5, 
                                        maximum=4, 
                                        value=1, 
                                        step=0.1, 
                                        label='⚡ Default Speech Speed'
                                    )
                                with gr.Column(scale=1):
                                    script_output_format = gr.Radio(
                                        choices=['WAV', 'MP3'],
                                        value='WAV',
                                        label='🎵 Output Format',
                                        info='Choose audio file format'
                                    )
                            
                            parse_btn = gr.Button('🔍 Parse Script & Detect Speakers', variant='secondary', size="lg")
                    
                    with gr.Column(scale=2):
                        # Voice assignment
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("<h3 style='margin-top: 0;'>🎤 Voice Assignment</h3>")
                            
                            # Dynamic speaker voice assignments (will be created after parsing)
                            speaker_voice_assignments = gr.State({})
                            detected_speakers = gr.State([])
                            
                            # Voice assignment interface (will be populated dynamically)
                            voice_assignment_interface = gr.Column(visible=False)
                            
                            with voice_assignment_interface:
                                speaker_voice_radios = []
                                for i in range(10):  # Support up to 10 speakers
                                    radio = gr.Radio(
                                        choices=get_voice_choices(),
                                        label=f"Speaker {i+1}",
                                        visible=False,
                                        interactive=True,
                                        value=None
                                    )
                                    speaker_voice_radios.append(radio)
                
                # Control buttons
                with gr.Row():
                    with gr.Column(scale=1):
                        generate_script_conversation_btn = gr.Button('🎬 Generate Conversation', variant='primary', size="lg")
                    with gr.Column(scale=1):
                        clear_script_btn = gr.Button('🗑️ Clear Script', variant='secondary', size="lg")
                
                # Output section
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("<h3 style='text-align: center; margin-top: 0;'>🎧 Generated Conversation</h3>")
                            conversation_audio = gr.Audio(
                                label=None, 
                                interactive=False, 
                                streaming=False, 
                                autoplay=False
                            )
                    
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("<h3 style='text-align: center; margin-top: 0;'>📋 Detected Script</h3>")
                            conversation_script = gr.Textbox(
                                label=None,
                                interactive=False,
                                lines=10,
                                placeholder="Parsed conversation will appear here..."
                            )
                
                # Tips section
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["card"]):
                        gr.Markdown("### ✅ Correct Format:")
                        gr.Textbox(
                            value="""Speaker 1: Hello there!
Speaker 2: Hi, how are you?
Alice: I'm doing great!
Bob: That's wonderful to hear.""",
                            label=None,
                            interactive=False,
                            lines=4,
                            max_lines=4
                        )
                    
                    with gr.Column(scale=1, elem_classes=["card"]):
                        gr.Markdown("### 🎯 Pro Tips:")
                        gr.Markdown("""
• Use "Speaker 1", "Speaker 2" or any names  
• Each line should start with "Name:"  
• Multi-line dialogue is supported  
• System auto-detects unique speakers  
• Perfect for scripts, dialogues, interviews  
                        """)

    with gr.Row(variant="panel", elem_id="footer-section"):
        gr.Markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; padding: 0.8rem 0;">
                <div style="margin: 8px 0;">
                    <p style="margin: 0 0 6px 0; font-size: 0.9rem; font-weight: 600;">🏷️ Voice Code Legend:</p>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 0.85rem;">
                        <span class="voice-code">af/am</span> <span>American English (Female/Male)</span>
                        <span class="voice-code">bf/bm</span> <span>British English (Female/Male)</span>
                        <span class="voice-code">IT</span> <span>Italian</span>
                        <span class="voice-code">pf/pm</span> <span>Brazilian Portuguese (Female/Male)</span>
                    </div>
                </div>
                <div style="margin: 8px 0;">
                    <p style="text-align: right; margin: 0; background: linear-gradient(90deg, #4776E6, #8E54E9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold; font-size: 1.1rem; animation: textGlow 3s infinite;">
                        Powered by Kokoro TTS
                    </p>
                    <p style="text-align: right; font-size: 0.8rem; margin: 3px 0 0 0; opacity: 0.8;">
                        Running 100% locally on your device
                    </p>
                </div>
            </div>
            """
        )

    # Connect buttons to functions
    generate_btn.click(fn=generate_first, inputs=[text, voice, speed, output_format], outputs=[out_audio, out_ps, large_file_info])
    
    # Connect file upload to voice assignment interface
    batch_files.change(
        fn=update_file_voice_assignments,
        inputs=[batch_files],
        outputs=[file_voice_assignments] + file_voice_radios
    )
    
    # Connect quick voice assignment button
    assign_same_voice_btn.click(
        fn=assign_same_voice_to_all,
        inputs=[quick_voice_select, batch_files],
        outputs=file_voice_radios
    )
    
    # Connect batch conversion functionality
    def handle_batch_conversion_with_voices(files, speed, output_format, *voice_assignments):
        summary, audio_files = batch_convert_text_files_with_voices(files, speed, output_format, *voice_assignments)
        audio_updates = update_batch_audio_players(audio_files)
        return [summary] + audio_updates
    
    batch_convert_btn.click(
        fn=handle_batch_conversion_with_voices,
        inputs=[batch_files, batch_speed, batch_output_format] + file_voice_radios,
        outputs=[batch_results, batch_audio_files] + batch_audio_players
    )
    
    # Update the voice list when refreshing
    def update_voice_list():
        updated_choices = update_voice_choices()
        return gr.update(choices=list(updated_choices.keys()), value=list(updated_choices.keys())[0])
    
    def update_all_voice_lists():
        updated_choices = update_voice_choices()
        voice_update = gr.update(choices=list(updated_choices.keys()), value=list(updated_choices.keys())[0])
        
        # Update all voice components
        updates = [voice_update]  # Main voice radio
        
        # Update file voice radios
        for _ in range(20):
            updates.append(gr.update(choices=list(updated_choices.keys())))
        
        # Update quick voice select
        updates.append(gr.update(choices=list(updated_choices.keys())[:5], value=list(updated_choices.keys())[0]))
        
        return updates
    
    refresh_btn.click(
        fn=update_all_voice_lists, 
        inputs=[], 
        outputs=[voice] + file_voice_radios + [quick_voice_select]
    )
    
    # Custom voice tab functionality
    upload_btn.click(
        fn=upload_and_refresh, 
        inputs=[custom_voice_files, custom_voice_name], 
        outputs=[upload_status, custom_voice_list, custom_voice_name, custom_voice_files]
    )
    
    refresh_custom_btn.click(fn=get_custom_voice_list, inputs=[], outputs=[custom_voice_list])

    # Connect voice mixer functionality
    # Connect all checkboxes and sliders to the formula builder
    all_inputs = []
    for checkbox, slider in zip(voice_checkboxes, voice_sliders):
        all_inputs.extend([checkbox, slider])
    
    # Update formula when any slider or checkbox changes
    for input_elem in all_inputs:
        input_elem.change(
            fn=build_formula_from_sliders,
            inputs=all_inputs,
            outputs=[voice_formula]
        )
    
    # Connect the mix button to generate the mixed voice
    mix_btn.click(
        fn=generate_mixed_voice,
        inputs=[voice_formula, mixed_voice_name, voice_text],
        outputs=[mix_status, mix_audio]
    )

    # Connect script-based conversation mode functionality
    # Parse script and show voice assignment interface
    parse_btn.click(
        fn=create_voice_assignment_interface,
        inputs=[conversation_script_input],
        outputs=[voice_assignment_interface] + speaker_voice_radios + [detected_speakers]
    )

    # Generate conversation from script with voice assignments
    generate_script_conversation_btn.click(
        fn=generate_from_script_with_voices,
        inputs=[conversation_script_input, script_pause_duration, script_speed, script_output_format] + speaker_voice_radios,
        outputs=[conversation_audio, conversation_script]
    )
    
    # Clear script conversation
    clear_script_btn.click(
        fn=clear_script_conversation,
        inputs=[],
        outputs=[conversation_script_input, conversation_audio, conversation_script, detected_speakers]
    )

    # Debug custom voices
    debug_custom_voices()

app.launch()
