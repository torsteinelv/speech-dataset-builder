import asyncio
import pyaudio
import time
from wyoming.audio import AudioChunk, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import SynthesizeStart, SynthesizeChunk, SynthesizeStop

class MyVoice:
    def __init__(self, name):
        self.name = name
        
    def to_dict(self):
        return {"name": self.name}

async def main():
    voice_name = "kathrine" 
    
    # Her kan du nå skrive hva du vil - kort eller langt!
    text_to_speak = "Hei, hvordan går det med deg i dag? Er du klar for å bytte skjermkort?"

    server_host = "10.10.0.33"
    server_port = 10200

    print(f"Kobler til Wyoming-server på {server_host}:{server_port}...")
    client = AsyncTcpClient(server_host, server_port)
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    try:
        async with client:
            voice_obj = MyVoice(voice_name)
            start_time = time.perf_counter()

            async def send_text():
                print("📤 Sender tekst...")
                await client.write_event(SynthesizeStart(voice=voice_obj).event())
                words = text_to_speak.replace("\n", " ").split(" ")
                
                for word in words:
                    if word.strip():
                        await client.write_event(SynthesizeChunk(text=word + " ").event())
                        await asyncio.sleep(0.05) # Litt raskere sending
                
                await client.write_event(SynthesizeStop().event())
                print("📤 Tekst ferdig sendt.")

            async def receive_audio():
                print("📥 Lytter etter lyd...")
                first_chunk = True
                is_playing = False
                audio_buffer = []
                buffer_limit = 15 # Maks buffer for lange tekster

                while True:
                    event = await client.read_event()
                    if event is None:
                        break

                    if AudioChunk.is_type(event.type):
                        if first_chunk:
                            elapsed = time.perf_counter() - start_time
                            print(f"\n⏱️ Respons: {elapsed:.2f}s")
                            first_chunk = False
                            
                        chunk = AudioChunk.from_event(event)
                        
                        if not is_playing:
                            audio_buffer.append(chunk.audio)
                            # Start avspilling hvis bufferen er full
                            if len(audio_buffer) >= buffer_limit:
                                is_playing = True
                                print(f"🔊 Buffer full ({buffer_limit} pakker). Starter avspilling...")
                                for b in audio_buffer:
                                    stream.write(b)
                                audio_buffer = []
                        else:
                            stream.write(chunk.audio)
                            
                    elif AudioStop.is_type(event.type):
                        # Tving avspilling av resten hvis vi ikke har startet enda
                        if not is_playing and audio_buffer:
                            print(f"🔊 Kort melding ferdig ({len(audio_buffer)} pakker). Spiller av...")
                            for b in audio_buffer:
                                stream.write(b)
                        break
                
                print(f"⏱️ Totaltid: {time.perf_counter() - start_time:.2f}s")

            await asyncio.gather(send_text(), receive_audio())

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main())
