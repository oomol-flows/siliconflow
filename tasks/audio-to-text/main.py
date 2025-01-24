from typing import Dict, Any
from oocana import Context
import requests
import logging
from pathlib import Path

# 配置常量
API_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
DEFAULT_MODEL = "FunAudioLLM/SenseVoiceSmall"
AUDIO_TYPE = "audio/wav"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(params: Dict[str, Any], context: Context) -> Dict[str, str]:
    """
    主函数：处理音频转文字任务
    
    Args:
        params: 包含音频文件路径和API密钥的字典
        context: Oocana上下文对象
        
    Returns:
        包含转换后文本的字典
        
    Raises:
        ValueError: 如果缺少必要参数
        RuntimeError: 如果转换失败
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
    调用API进行音频转文字
    
    Args:
        file_path: 音频文件路径
        model: 使用的模型名称
        api_key: API密钥
        
    Returns:
        API返回的JSON结果
        
    Raises:
        ValueError: 如果缺少API密钥
        requests.exceptions.RequestException: 如果API请求失败
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
