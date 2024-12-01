import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from .types import CringeRequest, CringeResponse
import uvicorn
from scripts.predictor import Predictor
import time
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class CringeService:
    def __init__(self):
        self.predictor = Predictor()
        self.request_queue = asyncio.Queue()
        self.task_processor_running = True
        self.request_counts = {}
        self.rate_limit = 10
        self.rate_window = 60
        self.executor = ThreadPoolExecutor(max_workers=4)

    def get_cringe_rating(self, score: float) -> str:
        if score < 0.2:
            return "Not cringe"
        elif score < 0.4:
            return "Somewhat cringe"
        elif score < 0.6:
            return "Cringe"
        elif score < 0.8:
            return "Very cringe"
        else:
            return "Extremely cringe"
    
    def _predict_sync(self, text: str) -> float:
        return self.predictor.predict(text) # This will run in thread pool
    
    async def predict_async(self, text: str) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, text) # Wrapper that will make that call in a thread pool haha

    async def process_request(self, text: str) -> CringeResponse:
        start_time = time.time()
        try:
            score = await self.predict_async(text)
            rating = self.get_cringe_rating(score)
            processing_time = time.time() - start_time

            return CringeResponse(
                score=score,
                rating=rating,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def check_rate_limit(self, ip_address: str) -> bool:
        current_time = time.time()
        if ip_address not in self.request_counts:
            self.request_counts[ip_address] = []
        self.request_counts[ip_address] = [t for t in self.request_counts[ip_address] if current_time - t < self.rate_window]
        if len(self.request_counts[ip_address]) >= self.rate_limit:
            return False
        self.request_counts[ip_address].append(current_time)
        return True
    
    async def cleanup(self):
        self.task_processor_running = False
        self.executor.shutdown(wait=True)


app = FastAPI()
cringe_service = CringeService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.post("/predict")
async def predict_cringe(request: Request, cringe_request: CringeRequest, background_tasks: BackgroundTasks):
    ip_address = request.client.host
    if not cringe_service.check_rate_limit(ip_address):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    try:
        body = await request.body()
        cringe_request = CringeRequest(**json.loads(body))
        response = await cringe_service.process_request(cringe_request.text)
        return response
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4, loop='uvloop')
    
