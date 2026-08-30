import asyncio
from src.async_client import AsyncDataFetcher

async def main():
    fetcher = AsyncDataFetcher(max_concurrency=3)
    user_ids = list(range(1, 6)) # Fetch users 1 to 5
    
    print(f"Fetching users {user_ids} concurrently...")
    users = await fetcher.batch_fetch_users(user_ids)
    
    print(f"\nSuccessfully validated {len(users)} user records:\n")
    for user in users:
        print(f" - ID {user.id}: {user.name} <{user.email}> | Company: {user.company_name} ({user.city})")

if __name__ == "__main__":
    asyncio.run(main())
