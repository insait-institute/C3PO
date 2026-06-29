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

"""Generic async-aware singleton pattern implementation.

Provides a ``Singleton`` base class that supports both synchronous and
asynchronous initialization. Subclasses inherit thread-safe, single-instance
guarantees with optional async setup via an ``async_init()`` method.
"""

import asyncio
from typing import Generic, Optional, TypeVar

import structlog

logger = structlog.stdlib.get_logger()

T = TypeVar('T')


# FIXME: type inferenced to Any
class Singleton(Generic[T]):
    """Generic base class implementing the singleton pattern with async support.

    Subclasses of ``Singleton`` are guaranteed to have at most one instance.
    Two initialization paths are provided:

    - **Async**: Use :meth:`get_instance_async` for classes that define an
      ``async_init()`` coroutine method. Initialization is protected by an
      ``asyncio.Lock`` for concurrency safety.
    - **Sync**: Use :meth:`get_instance_sync` for classes without async setup.
      Asserts that no ``async_init`` method exists to prevent misuse.

    Attributes:
        _instance: The singleton instance, or ``None`` if not yet created.
        _lock: An ``asyncio.Lock`` used for thread-safe async initialization.
    """

    _instance: Optional[T] = None
    _lock: Optional[asyncio.Lock] = None

    @classmethod
    async def get_instance_async(cls, *args, **kwargs):
        """Get or create the singleton instance asynchronously.

        On the first call, creates the instance, calls its ``async_init()``
        coroutine, and caches the result. Subsequent calls return the cached
        instance. Uses double-checked locking with an ``asyncio.Lock`` for
        concurrency safety.

        Args:
            *args: Positional arguments passed to the constructor on first init.
            **kwargs: Keyword arguments passed to the constructor on first init.

        Returns:
            The singleton instance.

        Raises:
            AssertionError: If the subclass does not define ``async_init()``.
        """
        if not cls._instance:
            if not cls._lock:
                cls._lock = asyncio.Lock()
            async with cls._lock:
                if not cls._instance:
                    self = cls(*args, **kwargs)
                    assert hasattr(self, 'async_init'), 'async singletons must define async_init function'
                    await self.async_init()
                    cls._instance = self
                    logger.debug('singleton class initialized', name=cls.__name__)
        return cls._instance

    @classmethod
    def get_instance_sync(cls, *args, **kwargs):
        """Get or create the singleton instance synchronously.

        On the first call, creates the instance and caches it. Subsequent
        calls return the cached instance. Asserts that the subclass does
        not define ``async_init()`` -- use :meth:`get_instance_async` for
        classes requiring async initialization.

        Args:
            *args: Positional arguments passed to the constructor on first init.
            **kwargs: Keyword arguments passed to the constructor on first init.

        Returns:
            The singleton instance.

        Raises:
            AssertionError: If the subclass defines ``async_init()`` (must use
                :meth:`get_instance_async` instead).
        """
        if not cls._instance:
            self = cls(*args, **kwargs)
            assert not hasattr(
                self, 'async_init'), f'class {cls.__name__} has async_init function, init it with get_instance_async.'
            cls._instance = self
            logger.debug('singleton class initialized', name=cls.__name__)
        return cls._instance
