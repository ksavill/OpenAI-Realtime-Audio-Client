import os
import threading
import websocket
import json
import base64
import pyaudio
import numpy as np

OPENAI_API_KEY = os.getenv("openai_token")

# Constants for audio recording and playback
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono
RATE = 24000  # 24kHz sampling rate
CHUNK = 1024  # Frame size

# WebSocket connection parameters
MODEL = 'gpt-4o-realtime-preview-2024-10-01'
URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta": "realtime=v1",
}

# Initialize PyAudio
p = pyaudio.PyAudio()

# Global variables
response_active = False  # Tracks if a response is currently active
audio_buffer = bytearray()  # Buffer to accumulate audio deltas

def float_to_int16(audio_data):
    """Convert numpy float32 array to int16."""
    return (audio_data * np.iinfo(np.int16).max).astype(np.int16)

def int16_to_float(audio_data):
    """Convert int16 audio data to float32."""
    return audio_data.astype(np.float32) / np.iinfo(np.int16).max

def audio_to_base64(audio_data):
    """Convert int16 audio data to base64-encoded string."""
    audio_bytes = audio_data.tobytes()
    return base64.b64encode(audio_bytes).decode('utf-8')

def base64_to_audio(audio_str):
    """Convert base64-encoded string to int16 audio data."""
    audio_bytes = base64.b64decode(audio_str)
    return np.frombuffer(audio_bytes, dtype=np.int16)

def on_message(ws, message):
    """Callback function when a message is received from the server."""
    global response_active, audio_buffer
    try:
        event = json.loads(message)
    except json.JSONDecodeError:
        print("Received non-JSON message.")
        return

    event_type = event.get('type')

    if not event_type:
        print("Received event without a type.")
        return

    # Handle various event types
    if event_type == 'response.audio.delta':
        audio_chunk = event.get('delta')
        if audio_chunk:
            audio_data = base64.b64decode(audio_chunk)
            audio_buffer.extend(audio_data)
        else:
            print("Warning: 'response.audio.delta' event received without 'delta' field.")

    elif event_type == 'response.audio.done':
        # Play the accumulated audio buffer
        if audio_buffer:
            audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
            play_audio(audio_array)
            audio_buffer = bytearray()  # Clear the buffer after playing
        else:
            print("Warning: 'response.audio.done' received but audio buffer is empty.")

    elif event_type == 'response.text.delta':
        text_chunk = event.get('delta')
        if text_chunk:
            print(f"Transcript delta: {text_chunk}", end='', flush=True)
        else:
            print("Warning: 'response.text.delta' event received without 'delta' field.")

    elif event_type == 'response.text.done':
        print("\nText response completed.")

    elif event_type == 'response.audio_transcript.delta':
        text_chunk = event.get('delta')
        if text_chunk:
            print(f"Audio transcript delta: {text_chunk}", end='', flush=True)
        else:
            print("Warning: 'response.audio_transcript.delta' event received without 'delta' field.")

    elif event_type == 'response.audio_transcript.done':
        print("\nAudio transcript completed.")

    elif event_type == 'response.created':
        print("Response created.")

    elif event_type == 'response.done':
        response_active = False  # Mark that the response has completed
        print("Response completed.")

    elif event_type == 'session.created':
        print("Session created.")

    elif event_type == 'error':
        error = event.get('error', {})
        print(f"Error: {error.get('message', 'Unknown error')}")

    # Handle other event types as needed
    else:
        print(f"Unhandled event type: {event_type}")

def on_error(ws, error):
    """Callback function when an error occurs."""
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    """Callback function when the WebSocket connection is closed."""
    print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

def on_open(ws):
    """Callback function when the WebSocket connection is established."""
    global response_active
    print("WebSocket connection opened")

    # Send the initial 'response.create' event to initialize the conversation
    response_create_event = {
        "type": "response.create",
        "response": {
            "modalities": ["text", "audio"],
            "instructions": "Please assist the user."
        }
    }
    ws.send(json.dumps(response_create_event))
    response_active = True  # Mark that a response is now active
    print("Sent initial 'response.create' event.")

    # Start a new thread to capture and send audio
    threading.Thread(target=send_audio_stream, args=(ws,), daemon=True).start()

def play_audio(audio_data):
    """Play int16 audio data."""
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True)
        stream.write(audio_data.tobytes())
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"Error during audio playback: {e}")

def send_audio_stream(ws):
    """Capture audio from the microphone and send it to the server."""
    global response_active
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording and sending audio. Press Ctrl+C to stop.")
    buffer_active = False  # Flag to track if audio has been sent since last commit
    try:
        while True:
            audio_frames = stream.read(CHUNK, exception_on_overflow=False)
            audio_base64 = base64.b64encode(audio_frames).decode('utf-8')

            # Send the audio chunk to the server
            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            ws.send(json.dumps(event))
            buffer_active = True  # Mark that audio has been sent

            # Commit the audio buffer periodically
            send_audio_stream.counter += 1
            if send_audio_stream.counter >= int(5 * RATE / CHUNK):
                if buffer_active:
                    commit_event = {"type": "input_audio_buffer.commit"}
                    ws.send(json.dumps(commit_event))
                    print("Committed audio buffer.")
                    send_audio_stream.counter = 0
                    buffer_active = False  # Reset buffer flag
                else:
                    # Buffer is empty; avoid committing
                    send_audio_stream.counter = 0

    except KeyboardInterrupt:
        print("Stopping audio capture.")
    except Exception as e:
        print(f"Exception in audio stream: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        ws.close()

# Initialize the counter attribute
send_audio_stream.counter = 0

def main():
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        URL,
        header=[f"{k}: {v}" for k, v in HEADERS.items()],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    # Run WebSocket in the main thread
    ws.run_forever()

if __name__ == "__main__":
    main()