from oocana import Context, data
from pathlib import Path
from openai import OpenAI
from typing import Any


def main(params: dict, context: Context):

    my_content = params.get("content")
    api_key = params.get("api_key")
    file_path = params.get("file_path")
    timbre = params.get("timbre")
    name = params.get("name")
    model = params.get("model")
    voice: Any = f"{model}:{timbre}"

    speech_file_path = f"{file_path}/{name}.mp3"
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=my_content,
        response_format="mp3",
    ) as response:
        response.stream_to_file(speech_file_path)
    context.preview({
        "type": "audio",
        "data": speech_file_path,
    })
    return {"audio_address": speech_file_path}
