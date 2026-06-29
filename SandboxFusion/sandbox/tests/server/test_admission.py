# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for run_code admission control (``_acquire_slot``).

These are pure asyncio unit tests of the slot-acquisition helper; they do
not need a running sandbox server.  They guard the behaviour that turns an
over-subscribed server's slow read-timeouts into fast, retryable 503s (see
documentation/apptainer-migration.md section 6.4).
"""

import asyncio
import time

import pytest
from fastapi import HTTPException

from sandbox.server.sandbox_api import _acquire_slot


async def test_no_semaphore_returns_false():
    """With concurrency limiting disabled there is no slot to release."""
    assert await _acquire_slot(None, queue_timeout=5) is False


async def test_acquires_free_slots_until_saturated():
    sem = asyncio.Semaphore(2)
    assert await _acquire_slot(sem, queue_timeout=1) is True
    assert await _acquire_slot(sem, queue_timeout=1) is True
    # Both permits now consumed; the next acquisition must be rejected fast.
    with pytest.raises(HTTPException) as exc:
        await _acquire_slot(sem, queue_timeout=0.1)
    assert exc.value.status_code == 503


async def test_rejects_with_503_when_saturated():
    """No free slot within queue_timeout -> fast HTTP 503, not a long wait."""
    sem = asyncio.Semaphore(1)
    await sem.acquire()  # server is now full

    start = time.monotonic()
    with pytest.raises(HTTPException) as exc:
        await _acquire_slot(sem, queue_timeout=0.2)
    elapsed = time.monotonic() - start

    assert exc.value.status_code == 503
    # Rejected at roughly the budget, and well before any client read timeout.
    assert 0.2 <= elapsed < 1.0, f'rejected after {elapsed:.2f}s'


async def test_waits_then_acquires_when_slot_frees_in_time():
    """A slot freed inside the budget is taken rather than rejected."""
    sem = asyncio.Semaphore(1)
    await sem.acquire()

    async def free_later():
        await asyncio.sleep(0.1)
        sem.release()

    asyncio.create_task(free_later())
    assert await _acquire_slot(sem, queue_timeout=5) is True


async def test_zero_timeout_waits_indefinitely():
    """queue_timeout<=0 keeps the original behaviour: wait for a slot, never 503."""
    sem = asyncio.Semaphore(1)
    await sem.acquire()

    async def free_later():
        await asyncio.sleep(0.3)
        sem.release()

    asyncio.create_task(free_later())
    # Would raise/return early if admission control mistakenly fired; the outer
    # wait_for only guards the test from hanging if the logic is broken.
    acquired = await asyncio.wait_for(_acquire_slot(sem, queue_timeout=0), timeout=5)
    assert acquired is True
