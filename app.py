from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import threading
import queue
import json
import asyncio

from main import Main  # adjust import if needed

app = FastAPI()
app.mount("/static", StaticFiles(directory="/home/ec2-user/PythonProject", html=True), name="static")
@app.get("/")
async def serve_index():
    return FileResponse("/home/ec2-user/PythonProject/index.html")

@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    await websocket.accept()

    try:
        # Receive the initial message from frontend
        data = await websocket.receive_text()
        payload = json.loads(data)
        question = payload.get("query")

        # Create a thread-safe queue
        q = queue.Queue()

        # Run your sync Spark pipeline in a separate thread
        def pipeline_worker():
            main = Main()
            main.run(question, q)

        thread = threading.Thread(target=pipeline_worker)
        thread.start()

        # Async loop that sends messages from queue to frontend
        while True:
            msg = q.get()

            print(f"[QUEUE SIZE] {q.qsize()}", flush=True)
            print(f"[MESSAGE] {msg}", flush=True)

            if msg.get("final"):
                await websocket.send_json({"final": True})
                break
            else:
                try:
                    await websocket.send_json(msg)
                    await asyncio.sleep(0.1)  # give frontend breathing room
                except Exception as e:
                    print(f"[!] send_json failed: {e}", flush=True)
                    break

    except Exception as e:
        print(f"[!] WebSocket handler error: {e}", flush=True)
        await websocket.send_json({"error": str(e)})

    finally:
        await websocket.close()
