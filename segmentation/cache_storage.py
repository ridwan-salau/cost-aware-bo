import fcntl
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import IO, Any, Callable, ContextManager, Iterator

import s3fs
from cachestore import LocalStorage
from cachestore.common import FileLock

s3 = s3fs.S3FileSystem()


class CustFileLock(FileLock):
    def acquire(self) -> None:
        if self._lockfile is None:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._lockfile = open(self._file_path, "w")
            fcntl.flock(self._lockfile, fcntl.LOCK_EX)


class AWSStorage(LocalStorage):
    def __init__(
        self,
        root: str | PathLike | None = None,
        openfn: Callable[..., IO[Any]] | Callable[..., ContextManager] | None = None,
    ) -> None:
        super().__init__(root, openfn)
        self._openfn = s3.open

    @contextmanager
    def open(self, key: str, mode: str) -> Iterator[IO[Any]]:
        filename = self._root / key
        lockfile = filename.parent / (filename.name + ".lock")
        s3.mkdirs(filename.parent, exist_ok=True)
        with CustFileLock(lockfile):
            try:
                with self._openfn(filename, mode) as fp:
                    yield fp
            except (Exception, KeyboardInterrupt):
                s3.rm_file(filename)
                raise
            finally:
                s3.rm_file(lockfile)

    def remove(self, key: str) -> None:
        filename = self._root / key
        s3.rm_file(filename)

    def exists(self, key: str) -> bool:
        return s3.exists(self._root / key)

    def all(self) -> Iterator[str]:
        for filename in s3.glob(self._root / "*"):
            yield Path(filename).name

    def filter(self, prefix: str) -> Iterator[str]:
        for filename in s3.glob(self._root / f"{prefix}*"):
            yield Path(filename).name
