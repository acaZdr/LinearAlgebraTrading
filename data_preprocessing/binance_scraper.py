import asyncio
import aiohttp
import os
import time
import ssl

from data_preprocessing.unzipper import run
from config import download_dir


async def download_url(session, url, file_path):
    try:
        # Bypass SSL verification by setting ssl=False
        async with session.get(url, ssl=False) as response:
            if response.status == 200:
                content = await response.read()
                with open(file_path, 'wb') as file:
                    file.write(content)
                return len(content)
    except ssl.SSLError as e:
        print(f"SSL Error for {url}: {e}")
    except aiohttp.ClientError as e:
        print(f"Client Error for {url}: {e}")
    except Exception as e:
        print(f"Unexpected error for {url}: {e}")
    return 0


async def download_files(start_month=8, start_year=17, end_month=5, end_year=24):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    start_time = time.time()
    total_size = 0

    async with aiohttp.ClientSession() as session:
        tasks = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                if y == end_year and m > end_month:
                    break
                if y == start_year and m < start_month:
                    continue

                m = f'0{m}' if m < 10 else str(m)
                file_name = f'BTCUSDT-1s-20{y}-{m}.zip'
                file_path = os.path.join(download_dir, file_name)

                if os.path.exists(file_path) or os.path.exists(file_path[:-4] + '.csv'):
                    print(f'File {file_name} already exists')
                    continue

                url = f'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1s/{file_name}'
                tasks.append(download_url(session, url, file_path))

        results = await asyncio.gather(*tasks)

        success_count = sum(1 for size in results if size > 0)
        total_size = sum(results)

    end_time = time.time()
    duration = end_time - start_time

    print(f'Successfully downloaded {success_count} out of {len(tasks)} files')
    print(f'Total download time: {duration:.2f} seconds')
    print(f'Total data downloaded: {total_size / (1024 * 1024):.2f} MB')
    print(f'Average download speed: {total_size / (1024 * 1024) / duration:.2f} MB/s')


def main():
    download_everything = input("Do you want to download all files? (y/n): ").lower() == 'y'
    if download_everything:
        asyncio.run(download_files())
    else:
        start_year = int(input("Enter the last two digits of the start year: "))
        start_month = int(input("Enter the start month: "))
        end_year = int(input("Enter the last two digits of the end year: "))
        end_month = int(input("Enter the end month: "))
        asyncio.run(download_files(start_month, start_year, end_month, end_year))
    run()


if __name__ == '__main__':
    main()
