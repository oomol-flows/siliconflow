from typing import Dict, Any
from oocana import Context
import requests
import logging
from pathlib import Path

#region generated meta
import typing
class Inputs(typing.TypedDict):
    audio: str
    api_key: str
    model: typing.Literal["FunAudioLLM/SenseVoiceSmall"]
class Outputs(typing.TypedDict):
    text: str
#endregion

# Configuration constants
API_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
DEFAULT_MODEL = "FunAudioLLM/SenseVoiceSmall"
AUDIO_TYPE = "audio/wav"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(params: Dict[str, Any], context: Context) -> Dict[str, str]:
    """
    Main function: Handles audio to text transcription task
    
    Args:
        params: Dictionary containing audio file path and API key
        context: Oocana context object
        
    Returns:
        Dictionary containing converted text
        
    Raises:
        ValueError: If required parameters are missing
        RuntimeError: If conversion fails
    """
    audio = params.get('audio')
    api_key = params.get('api_key')
    
    if not all([audio, api_key]):
        error_msg = "Missing required parameters: audio and api_key are required"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # 类型检查
    assert isinstance(audio, str), "audio parameter must be a string"
    assert isinstance(api_key, str), "api_key parameter must be a string"
    
    try:
        result = transcribe_audio(audio, DEFAULT_MODEL, api_key)
        out_text = result.get("text", "")
        
        if not out_text:
            warning_msg = "Transcription returned empty text"
            logger.warning(warning_msg)
            
        context.preview({
            "type": "markdown",
            "data": out_text,
        })
        
        return {"text": out_text}
        
    except Exception as e:
        error_msg = f"Audio transcription failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def transcribe_audio(file_path: str, model: str, api_key: str) -> Dict[str, Any]:
    """
    Calls API for audio to text transcription
    
    Args:
        file_path: Audio file path
        model: Model name to use
        api_key: API key
        
    Returns:
        JSON result from API
        
    Raises:
        ValueError: If API key is missing
        requests.exceptions.RequestException: If API request fails
    """
    if not api_key:
        error_msg = "API key is required"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not Path(file_path).exists():
        error_msg = f"Audio file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        with open(file_path, 'rb') as file:
            files = [
                ('file', (file.name, file, AUDIO_TYPE)),
                ('model', (None, model))
            ]
            
            response = requests.post(API_URL, headers=headers, files=files)
            response.raise_for_status()
            
            return response.json()
            
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        logger.error(error_msg)
        raise
