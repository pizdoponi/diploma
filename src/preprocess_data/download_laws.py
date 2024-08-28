import asyncio
import os
from argparse import ArgumentParser

from playwright.async_api import async_playwright


async def download_page(url, browser, directory="laws"):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = url.split("/")[-1] + ".html"
    file_path = os.path.join(directory, file_name)

    # Only download if the file doesn't already exist
    if not os.path.isfile(file_path):
        # Create a new page
        page = await browser.new_page()
        try:
            # Go to the URL and wait until network is idle
            await page.goto(url, wait_until="networkidle")
            # Save the content of the page
            content = await page.content()
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
            print(f"Downloaded and saved: {file_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
        finally:
            await page.close()
    else:
        print(f"File already exists: {file_path}")


async def process_urls(urls, directory):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        tasks = [download_page(url, browser, directory) for url in urls]
        await asyncio.gather(*tasks)
        await browser.close()


def read_urls_from_file(file_path):
    with open(file_path, "r") as file:
        return [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]


async def main(args):
    if args.url:
        await process_urls(args.url, "laws")
    elif args.file:
        urls = read_urls_from_file(args.file)
        await process_urls(urls, "laws")
    else:
        print(
            "Invalid arguments: No URLs or file specified. Use -u for URLs or -f for a file, or see --help."
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Download one or more slovenian laws by providing URLs or a file with URLs. Each URL should point to a glasilo uradnega lista RS, that is to the actual page where html is served (for that law only). For example: https://www.uradni-list.si/glasilo-uradni-list-rs/vsebina/2021-01-3972."
    )
    parser.add_argument(
        "-u", "--url", type=str, nargs="+", help="Specify one or more URLs."
    )
    parser.add_argument(
        "-f", "--file", type=str, help="Specify a file containing URLs, one per line."
    )

    args = parser.parse_args()
    asyncio.run(main(args))
