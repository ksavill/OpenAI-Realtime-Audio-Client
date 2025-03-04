import os
import threading
import websocket
import json
import base64
import pyaudio
import numpy as np
import queue
from websocket._exceptions import WebSocketConnectionClosedException

class RealtimeClient:
    def __init__(self):
        """Initialize the RealtimeClient with audio and WebSocket configurations."""
        self.p = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 24000
        self.CHUNK = 1024
        self.BUFFER_THRESHOLD = 4800  # ~100ms of audio
        self.audio_queue = queue.Queue()
        self.audio_buffer = bytearray()
        self.response_active = False
        self.api_key = os.getenv("openai_token") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found in 'openai_token' or 'OPENAI_API_KEY' environment variables.")
        self.MODEL = 'gpt-4o-realtime-preview-2024-10-01'
        self.URL = f"wss://api.openai.com/v1/realtime?model={self.MODEL}"
        self.HEADERS = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

    def start(self):
        """Start the WebSocket connection, audio playback, and text input handling."""
        self.ws = websocket.WebSocketApp(
            self.URL,
            header=[f"{k}: {v}" for k, v in self.HEADERS.items()],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        threading.Thread(target=self.play_audio_thread, daemon=True).start()
        threading.Thread(target=self.ws.run_forever, daemon=True).start()
        self.send_text_input()

    def play_audio_thread(self):
        """Play audio from the queue in a separate thread."""
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, output=True)
        try:
            while True:
                audio_data = self.audio_queue.get()
                if audio_data is None:
                    break
                try:
                    stream.write(audio_data.tobytes())
                except Exception as e:
                    print(f"Audio playback error: {e}")
                self.audio_queue.task_done()
        finally:
            stream.stop_stream()
            stream.close()

    def send_audio_stream(self):
        """Stream audio input to the WebSocket when voice activity is detected."""
        device_index = self.get_default_input_device()
        if device_index is None:
            print("Audio streaming disabled due to no microphone.")
            return

        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK
        )
        print("Recording audio. Type 'exit' in CLI to stop.")

        HISTORY_SIZE = 50
        THRESHOLD_MULTIPLIER = 1.5
        MIN_THRESHOLD = 500
        BUFFER_SIZE = 5

        rms_history = []
        audio_chunk_buffer = []

        try:
            while True:
                try:
                    audio_frames = stream.read(self.CHUNK, exception_on_overflow=False)
                except IOError as e:
                    print(f"Audio read error: {e}")
                    continue

                audio_data = np.frombuffer(audio_frames, dtype=np.int16)
                if len(audio_data) == 0:
                    continue
                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))

                rms_history.append(rms)
                if len(rms_history) > HISTORY_SIZE:
                    rms_history.pop(0)
                avg_rms = sum(rms_history) / len(rms_history) if rms_history else 0
                dynamic_threshold = max(avg_rms * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)

                if rms > dynamic_threshold:
                    audio_chunk_buffer.append(audio_frames)
                    if len(audio_chunk_buffer) >= BUFFER_SIZE:
                        for chunk in audio_chunk_buffer:
                            event = {
                                "type": "input_audio_buffer.append",
                                "audio": self.audio_to_base64(np.frombuffer(chunk, dtype=np.int16))
                            }
                            try:
                                self.ws.send(json.dumps(event))
                            except WebSocketConnectionClosedException:
                                print("WebSocket closed, stopping audio stream.")
                                return
                        audio_chunk_buffer = []
                else:
                    if audio_chunk_buffer and len(audio_chunk_buffer) * self.CHUNK >= 2400:
                        try:
                            self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            response_event = {
                                "type": "response.create",
                                "response": {
                                    "modalities": ["text", "audio"]
                                }
                            }
                            self.ws.send(json.dumps(response_event))
                        except WebSocketConnectionClosedException:
                            print("WebSocket closed, stopping audio stream.")
                            return
                        audio_chunk_buffer = []
        except Exception as e:
            print(f"Audio stream error: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    def send_text_input(self):
        """Handle text input from the console and send it to the WebSocket."""
        print("Enter text (or 'exit' to quit):")
        while True:
            text = input("> ")
            if text.lower() == 'exit':
                self.audio_queue.put(None)
                self.ws.close()
                break

            item_create_event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        }
                    ]
                }
            }
            try:
                self.ws.send(json.dumps(item_create_event))
                response_event = {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"]
                    }
                }
                self.ws.send(json.dumps(response_event))
            except WebSocketConnectionClosedException:
                print("WebSocket closed, cannot send text input.")
                break

    def on_open(self, ws):
        """Initialize the WebSocket session and start audio streaming."""
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": "Assist the user with both text and audio input. Avoid using any banned words."
            }
        }
        self.ws.send(json.dumps(session_update))
        threading.Thread(target=self.send_audio_stream, daemon=True).start()

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages, logging only errors and text responses."""
        try:
            event = json.loads(message)
            event_type = event.get('type')
        except json.JSONDecodeError:
            print("Received non-JSON message.")
            return

        if event_type == 'response.text.delta':
            text_chunk = event.get('delta')
            if text_chunk:
                print(f"{text_chunk}", end='', flush=True)
        elif event_type == 'error':
            error = event.get('error', {})
            print(f"Error: {error.get('message', 'Unknown error')}")
        elif event_type == 'response.audio.delta':
            audio_chunk = event.get('delta')
            if audio_chunk:
                audio_data = base64.b64decode(audio_chunk)
                self.audio_buffer.extend(audio_data)
                while len(self.audio_buffer) >= self.BUFFER_THRESHOLD:
                    chunk_to_play = self.audio_buffer[:self.BUFFER_THRESHOLD]
                    audio_array = np.frombuffer(chunk_to_play, dtype=np.int16)
                    self.audio_queue.put(audio_array)
                    self.audio_buffer = self.audio_buffer[self.BUFFER_THRESHOLD:]
        elif event_type == 'response.audio.done':
            if self.audio_buffer:
                audio_array = np.frombuffer(self.audio_buffer, dtype=np.int16)
                self.audio_queue.put(audio_array)
                self.audio_buffer = bytearray()

    def on_error(self, ws, error):
        """Log WebSocket errors."""
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Log WebSocket closure."""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")

    def get_default_input_device(self):
        """Find the default input device index."""
        device_count = self.p.get_device_count()
        for i in range(device_count):
            device_info = self.p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels', 0) > 0:
                return i
        return None

    @staticmethod
    def audio_to_base64(audio_data):
        """Convert int16 audio data to base64-encoded string."""
        return base64.b64encode(audio_data.tobytes()).decode('utf-8')


if __name__ == "__main__":
    client = RealtimeClient()
    client.start()