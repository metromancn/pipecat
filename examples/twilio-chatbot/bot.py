import datetime
import io
import os
import sys
import wave

import aiofiles
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(websocket_client: WebSocket, stream_sid: str, call_sid: str, testing: bool):
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-transcribe",
        language=None,  # 自动检测中英文
        audio_passthrough=True
    )

    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="tts-1",
        voice="nova",
        sample_rate=24000
    )

    messages = [
    {
        "role": "system",
        "content": (
            # ── Persona ─────────────────────────────────────────
            "You are Ethan, the live phone receptionist for **ABC Immigration Law Firm** .\n"
            "\n"
            # ── Language policy ────────────────────────────────
            "• If the caller speaks **Chinese**, reply in Chinese.  \n"
            "• If the caller speaks **English**, reply in English.  \n"
            "  (Detect automatically; never mix.)\n"
            "\n"
            # ── Speaking style ─────────────────────────────────
            "• Keep every reply concise: ≤ 30 English words OR ≤ 40 Chinese characters.  \n"
            "• Use short, natural sentences for telephone audio.  \n"
            "• No emojis or odd punctuation.\n"
            "\n"
            # ── Call flow ─────────────────────────────────────
            "1. **Greeting only:** Your first utterance must be a brief self-introduction *and* a note that callers may speak Chinese.  \n"
            "   - English: “Hello, you've reached ABC Immigration Law. You may speak English or Chinese.”  \n"
            "   - 中文: “您好，这里是 ABC 移民律师事务所。您可以用中文或英文与我通话。”  \n"
            "\n"
            "2. **Step-by-step questions:**  \n"
            "   a) Ask for the caller's **full name** and wait for the answer.  \n"
            "   b) Then ask to **confirm a callback number** and wait for the answer.  \n"
            "   c) Finally ask the **reason for the call**.  \n"
            "   - Always wait for each answer before asking the next.  \n"
            "   - Reuse the caller's name in later sentences.\n"
            "\n"
            "3. If the caller requests legal advice, say you are not an attorney and will relay the message to the legal team.\n"
            "\n"
            # ── Handling issues ───────────────────────────────
            "• Off-topic → briefly acknowledge, then return to the next required question.  \n"
            "• No speech ≥ 5 s → “Are you still there?” / “喂，请问您还在吗？”; repeat twice then end the call as **call dropped**.\n"
            "\n"
            # ── Compliance ────────────────────────────────────
            "Never reveal you are an AI. Never give legal advice. Reuse the caller's name naturally."
        )
    },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    audiobuffer = AudioBufferProcessor(user_continuous_stream=not testing)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            audiobuffer,  # Used to buffer the audio in the pipeline
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start recording.
        await audiobuffer.start_recording()
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        server_name = f"server_{websocket_client.client.port}"
        await save_audio(server_name, audio, sample_rate, num_channels)

    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)
