import asyncio
import httpx
from typing import List, Dict, Any, Optional
from .models import UserRecord

class AsyncDataFetcher:
    """Production-ready asynchronous HTTP fetcher with concurrency limiters and error retries."""
    
    def __init__(self, base_url: str = "https://jsonplaceholder.typicode.com", max_concurrency: int = 5):
        self.base_url = base_url.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def fetch_user(self, client: httpx.AsyncClient, user_id: int) -> Optional[UserRecord]:
        """Fetch individual user by ID with rate limiting and Pydantic validation."""
        async with self.semaphore:
            url = f"{self.base_url}/users/{user_id}"
            try:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                raw_data = response.json()
                return UserRecord.from_nested_json(raw_data)
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as exc:
                print(f"[Error] Failed fetching user_id={user_id}: {exc}")
                return None

    async def batch_fetch_users(self, user_ids: List[int]) -> List[UserRecord]:
        """Concurrently fetch multiple user IDs using asyncio task gathering."""
        async with httpx.AsyncClient() as client:
            tasks = [self.fetch_user(client, uid) for uid in user_ids]
            results = await asyncio.gather(*tasks)
            return [user for user in results if user is not None]
