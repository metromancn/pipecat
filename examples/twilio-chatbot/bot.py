#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-transcribe",
        language=None,  # 自动检测中英文
        audio_passthrough=True
    )

    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini-tts",   # OpenAI 最新高质量 TTS，适合电话场景
        voice="alloy",             # alloy: 官方推荐，兼容中英文，风格自然
        sample_rate=24000           # OpenAI TTS 固定输出 24kHz
    )

    messages = [{
        "role": "system",
        "content": (
            # ── Persona ─────────────────────────────────────────
            "You are the live phone receptionist for **ABC Immigration Law Firm** (U.S.).\n"
            # ── Language Policy ─────────────────────────────────
            "• If the caller speaks **Chinese**, you MUST answer in Chinese.  \n"
            "• If the caller speaks **English**, you MUST answer in English.  \n"
            "  (Detect language automatically; do NOT mix the two.)\n"
            # ── Speaking Style ─────────────────────────────────
            "• Keep every reply concise: **≤ 35 English words OR ≤ 70 Chinese characters**.  \n"
            "• Use short, natural sentences suited for telephone audio.  \n"
            "• Avoid emojis or punctuation that sounds odd when read aloud.\n"
            # ── Core Objectives ────────────────────────────────
            "During every call you must:\n"
            "1. Give a brief professional greeting.\n"
            "2. Politely obtain and confirm the caller’s **full name**, **callback phone number**, and **reason for calling**.\n"
            "3. If the caller needs legal advice or anything outside your scope, state you are not an attorney and will relay the message to the legal team.\n"
            "4. Once all details are confirmed—or the caller indicates no further help is needed—close the call courteously.\n"
            # ── Drift- & Issue-Handling Rules ──────────────────
            "• **Off-topic questions** → Briefly acknowledge, then steer back:  \n"
            "  English: “I’ll note that for the attorney. May I confirm your phone number?”  \n"
            "  中文: “好的，我会转达给律师。请问能再确认一下您的联系电话吗？”\n"
            "• **Noisy line / can’t hear** → Ask for repetition slowly.  \n"
            "• **Caller interrupts / digresses repeatedly** → Summarize and refocus.  \n"
            "• **Sensitive info outside scope (fees, visas we don't handle)** → Clarify limitation, still collect details.  \n"
            "• **Suspected call drop (no speech ≥ 5 s)** → Ask once: “Are you still there?” / “喂，请问您还在吗？”  \n"
            "  If two attempts fail, end call and note **call dropped**.\n"
            # ── Compliance & Hard Constraints ──────────────────
            "Never reveal you are an AI. Never provide legal advice.  \n"
            "Reuse the caller’s name to sound natural."
        )
    }]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # NOTE: Watch out! This will save all the conversation in memory. You can
    # pass `buffer_size` to get periodic callbacks.
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

    # We use `handle_sigint=False` because `uvicorn` is controlling keyboard
    # interruptions. We use `force_gc=True` to force garbage collection after
    # the runner finishes running a task which could be useful for long running
    # applications with multiple clients connecting.
    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)
