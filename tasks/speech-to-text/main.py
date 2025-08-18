from oocana import Context
from openai import OpenAI

#region generated meta
import typing
class Inputs(typing.TypedDict):
    audio: str
    api_key: str
    model: typing.Literal["FunAudioLLM/SenseVoiceSmall"]
class Outputs(typing.TypedDict):
    text: str
#endregion

def main(params: Inputs, context: Context) -> Outputs:
    """
    Main function: Handles audio to text transcription task
    """
    client = OpenAI(
        api_key=params["api_key"],
        base_url="https://api.siliconflow.cn/v1"
    )
    
    with open(params["audio"], "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=params.get("model", "FunAudioLLM/SenseVoiceSmall"),
            file=audio_file
        )
    
    context.preview({
        "type": "markdown",
        "data": transcription.text,
    })
    
    return {"text": transcription.text}
