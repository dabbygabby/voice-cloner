import os
import sqlite3
import uuid
import shutil
from datetime import datetime
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional, List
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CTRANSLATE2_CUDA_ALLOCATOR"] = "0"

# Import OpenVoice components
import mps_patch  # Apply MPS patch first

# Additional environment setup for Apple Silicon
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Force disable MPS for CTranslate2 compatibility
    torch.backends.mps.is_available = lambda: False

from openvoice import se_extractor
from openvoice.api import ToneColorConverter, BaseSpeakerTTS

app = FastAPI(title="OpenVoice V2 API Server", version="2.0.0")

# Configuration
CHECKPOINT_DIR = os.environ.get("OPENVOICE_CHECKPOINT_DIR", "checkpoints_v2")
VOICES_DIR = "voices"
AUDIO_DIR = "audio_files"
DATABASE_PATH = "voice_database.db"

# Ensure directories exist
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

def ensure_checkpoints(checkpoint_dir: str) -> None:
    """Ensure required OpenVoice V2 checkpoints exist; download and extract if missing."""
    required_paths = [
        f"{checkpoint_dir}/converter/config.json",
        f"{checkpoint_dir}/converter/checkpoint.pth",
        # V2 ships source speaker embeddings under base_speakers/ses
        f"{checkpoint_dir}/base_speakers/ses/en-newest.pth",
    ]
    if all(os.path.exists(p) for p in required_paths):
        return

    os.makedirs(checkpoint_dir, exist_ok=True)
    download_url = (
        "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
    )
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        tmp_zip_path = tmp_zip.name
    try:
        urllib.request.urlretrieve(download_url, tmp_zip_path)
        with zipfile.ZipFile(tmp_zip_path, 'r') as zf:
            # Extract to a temp directory first
            with tempfile.TemporaryDirectory() as tmp_dir:
                zf.extractall(tmp_dir)
                # Find extracted checkpoints folder
                candidate_root = None
                for root, dirs, files in os.walk(tmp_dir):
                    if (
                        os.path.exists(os.path.join(root, 'converter', 'config.json'))
                        and os.path.exists(os.path.join(root, 'converter', 'checkpoint.pth'))
                    ):
                        candidate_root = root
                        break
                if candidate_root is None:
                    raise RuntimeError("Downloaded checkpoints archive does not contain expected structure")
                # Move/merge into checkpoint_dir
                for item in os.listdir(candidate_root):
                    src = os.path.join(candidate_root, item)
                    dst = os.path.join(checkpoint_dir, item)
                    if os.path.isdir(src):
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
                    else:
                        shutil.move(src, dst)
    finally:
        if os.path.exists(tmp_zip_path):
            os.unlink(tmp_zip_path)

def ensure_base_checkpoints_v1() -> str:
    """Ensure V1 base TTS checkpoints exist; returns the base dir path (e.g., 'checkpoints')."""
    base_dir = "checkpoints"
    req_files = [
        f"{base_dir}/base_speakers/EN/config.json",
        f"{base_dir}/base_speakers/EN/checkpoint.pth",
    ]
    if all(os.path.exists(p) for p in req_files):
        return base_dir
    os.makedirs(base_dir, exist_ok=True)
    url = "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip"
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        tmp_zip_path = tmp_zip.name
    try:
        urllib.request.urlretrieve(url, tmp_zip_path)
        with zipfile.ZipFile(tmp_zip_path, 'r') as zf:
            with tempfile.TemporaryDirectory() as tmp_dir:
                zf.extractall(tmp_dir)
                # The archive should contain 'checkpoints' root
                candidate_root = None
                for root, dirs, files in os.walk(tmp_dir):
                    if (
                        os.path.exists(os.path.join(root, 'base_speakers', 'EN', 'config.json'))
                        and os.path.exists(os.path.join(root, 'base_speakers', 'EN', 'checkpoint.pth'))
                    ):
                        candidate_root = root
                        break
                if candidate_root is None:
                    raise RuntimeError("Downloaded V1 checkpoints archive missing EN base TTS")
                for item in os.listdir(candidate_root):
                    src = os.path.join(candidate_root, item)
                    dst = os.path.join(base_dir, item)
                    if os.path.isdir(src):
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
                    else:
                        shutil.move(src, dst)
    finally:
        if os.path.exists(tmp_zip_path):
            os.unlink(tmp_zip_path)
    return base_dir


# Ensure checkpoints are present before initializing models
ensure_checkpoints(CHECKPOINT_DIR)

# Initialize OpenVoice components
ckpt_converter = f'{CHECKPOINT_DIR}/converter'
device = "cpu"  # Force CPU for Apple Silicon compatibility
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Accent to SES mapping (files exist under base_speakers/ses)
ACCENT_TO_SES_FILE = {
    'en-au': 'en-au.pth',
    'en-br': 'en-br.pth',
    'en-default': 'en-default.pth',
    'en-india': 'en-india.pth',
    'en-newest': 'en-newest.pth',
    'en-us': 'en-us.pth',
    'es': 'es.pth',
    'fr': 'fr.pth',
    'jp': 'jp.pth',
    'kr': 'kr.pth',
    'zh': 'zh.pth',
}

# Database models
class VoiceModel(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    file_path: str
    created_at: str

class TTSRequest(BaseModel):
    text: str
    voice_id: str
    accent: str = "en-newest"
    speed: float = 1.0

# Database setup
def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            file_path TEXT NOT NULL,
            se_file_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_audio (
            id TEXT PRIMARY KEY,
            voice_id TEXT NOT NULL,
            text TEXT NOT NULL,
            file_path TEXT NOT NULL,
            accent TEXT,
            speed REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (voice_id) REFERENCES voice_models (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

@app.get("/")
async def root():
    return {"message": "OpenVoice V2 API Server", "version": "2.0.0"}

@app.post("/upload_voice/")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None)
):
    """Upload a voice sample for cloning"""
    
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Generate unique ID
    voice_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix or '.wav'
    voice_file_path = f"{VOICES_DIR}/{voice_id}{file_extension}"
    se_file_path = f"{VOICES_DIR}/{voice_id}.pth"
    
    with open(voice_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Extract speaker embedding
        reference_speaker, _audio_name = se_extractor.get_se(
            voice_file_path,
            tone_color_converter,
            vad=False,
        )
        
        # Save speaker embedding
        import torch
        torch.save(reference_speaker, se_file_path)
        
        # Save to database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO voice_models (id, name, description, file_path, se_file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (voice_id, name, description, voice_file_path, se_file_path, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return {
            "voice_id": voice_id,
            "name": name,
            "description": description,
            "message": "Voice uploaded and processed successfully"
        }
        
    except Exception as e:
        # Clean up files if processing failed
        if os.path.exists(voice_file_path):
            os.remove(voice_file_path)
        if os.path.exists(se_file_path):
            os.remove(se_file_path)
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

@app.get("/voices/")
async def list_voices():
    """List all available voice models"""
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name, description, created_at FROM voice_models')
    voices = cursor.fetchall()
    
    conn.close()
    
    return [
        {
            "id": voice[0],
            "name": voice[1],
            "description": voice[2],
            "created_at": voice[3]
        }
        for voice in voices
    ]

@app.post("/synthesize/")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech using a cloned voice"""
    
    # Get voice model from database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT se_file_path FROM voice_models WHERE id = ?', (request.voice_id,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Voice model not found")
    
    se_file_path = result[0]
    
    try:
        # Load speaker embedding
        import torch
        target_se = torch.load(se_file_path, map_location=device)
        
        # Generate unique filename for output
        output_id = str(uuid.uuid4())
        output_path = f"{AUDIO_DIR}/{output_id}.wav"
        
        # Create temporary file for base TTS
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Determine base TTS config/ckpt: prefer V2 EN_V2 if present, else fallback to V1 EN
        v2_env2_cfg = f'{CHECKPOINT_DIR}/base_speakers/EN_V2/config.json'
        v2_env2_ckpt = f'{CHECKPOINT_DIR}/base_speakers/EN_V2/checkpoint.pth'
        if os.path.exists(v2_env2_cfg) and os.path.exists(v2_env2_ckpt):
            base_cfg, base_ckpt = v2_env2_cfg, v2_env2_ckpt
        else:
            v1_base_dir = ensure_base_checkpoints_v1()
            base_cfg = f'{v1_base_dir}/base_speakers/EN/config.json'
            base_ckpt = f'{v1_base_dir}/base_speakers/EN/checkpoint.pth'

        # Initialize base TTS
        model = BaseSpeakerTTS(base_cfg, device=device)
        model.load_ckpt(base_ckpt)
        
        # Map accent to source speaker embedding file
        ses_file = ACCENT_TO_SES_FILE.get(request.accent, 'en-newest.pth')
        source_se_path = f'{CHECKPOINT_DIR}/base_speakers/ses/{ses_file}'
        import torch as _torch
        source_se = _torch.load(source_se_path, map_location=device)
        
        # Render base TTS audio to tmp_path using an available speaker key
        # Prefer 'EN' if present, else first available key
        speakers = list(model.hps.speakers.keys()) if hasattr(model.hps, 'speakers') else []
        speaker_key = 'EN' if 'EN' in speakers else (speakers[0] if speakers else 'EN')
        model.tts(request.text, tmp_path, speaker_key, language='English', speed=request.speed)
        
        # Apply voice conversion
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=tmp_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            message=encode_message
        )
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Save generation record to database
        cursor.execute('''
            INSERT INTO generated_audio (id, voice_id, text, file_path, accent, speed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (output_id, request.voice_id, request.text, output_path, request.accent, request.speed, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return {
            "audio_id": output_id,
            "voice_id": request.voice_id,
            "text": request.text,
            "file_path": output_path,
            "message": "Speech synthesized successfully"
        }
        
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Download generated audio file"""
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT file_path FROM generated_audio WHERE id = ?', (audio_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if not result or not os.path.exists(result[0]):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(result[0], media_type="audio/wav", filename=f"{audio_id}.wav")

@app.get("/history/")
async def get_generation_history():
    """Get history of generated audio"""
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT ga.id, ga.voice_id, vm.name as voice_name, ga.text, ga.accent, ga.speed, ga.created_at
        FROM generated_audio ga
        JOIN voice_models vm ON ga.voice_id = vm.id
        ORDER BY ga.created_at DESC
    ''')
    
    history = cursor.fetchall()
    conn.close()
    
    return [
        {
            "audio_id": record[0],
            "voice_id": record[1],
            "voice_name": record[2],
            "text": record[3],
            "accent": record[4],
            "speed": record[5],
            "created_at": record[6]
        }
        for record in history
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)