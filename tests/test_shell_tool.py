import pytest
import asyncio


@pytest.mark.asyncio
async def test_async_shell():
    process = await asyncio.create_subprocess_shell(
        cmd="ls -la",
        shell=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
