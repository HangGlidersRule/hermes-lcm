"""Regression: opening a MessageStore must not drop SQLite's POSIX locks.

On POSIX, closing ANY descriptor for a file releases every fcntl lock the process
holds on that file, including locks taken by SQLite connections that are already
open (https://sqlite.org/howtocorrupt.html, section 2.2). The storage-permission
helpers used to open and close their own descriptors on the live database and its
sidecars every time a MessageStore was created (the gateway creates one per agent
clone). With the locks gone, another process treats the live database as idle:
when it closes it deletes the WAL, it rebuilds the shared-memory index, or it
checkpoints and shrinks the file under the gateway's mmap. Committed rows vanish,
pages go "malformed", and the gateway dies with SIGBUS.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_lcm.store import MessageStore

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX advisory locks")

# SQLite's WAL "dead man switch" lock: byte 128 of the -shm file
# (WALINDEX_LOCK_OFFSET 120 + 8). Every connection holds it shared; a connection
# that can take it exclusively concludes it is the only one using the database.
_SHM_DMS_OFFSET = 128

_PROBE = r"""
import fcntl, os, sys
path, offset = sys.argv[1], int(sys.argv[2])
fd = os.open(path, os.O_RDWR)
try:
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, offset, 0)
except OSError:
    print("held")
else:
    fcntl.lockf(fd, fcntl.LOCK_UN, 1, offset, 0)
    print("free")
finally:
    os.close(fd)
"""


def _lock_seen_by_another_process(path: Path, offset: int) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(path), str(offset)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_opening_a_second_store_keeps_the_first_connections_locks(tmp_path):
    db_path = tmp_path / "lcm.db"
    shm_path = db_path.with_name(db_path.name + "-shm")
    first = MessageStore(db_path)
    try:
        first.append("lock-retention", {"role": "user", "content": "keep my locks"})
        first.commit()
        assert _lock_seen_by_another_process(shm_path, _SHM_DMS_OFFSET) == "held"

        # A per-agent clone opens its own store on the same database in this process.
        second = MessageStore(db_path)
        try:
            assert _lock_seen_by_another_process(shm_path, _SHM_DMS_OFFSET) == "held"
        finally:
            second.close()

        assert _lock_seen_by_another_process(shm_path, _SHM_DMS_OFFSET) == "held"
        assert first.connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
    finally:
        first.close()


_OTHER_PROCESS_READER = r"""
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("SELECT count(*) FROM messages").fetchone()
conn.close()
"""


def test_another_process_closing_the_database_keeps_the_wal(tmp_path):
    db_path = tmp_path / "lcm.db"
    wal_path = db_path.with_name(db_path.name + "-wal")
    first = MessageStore(db_path)
    try:
        first.append("lock-retention", {"role": "user", "content": "still in the WAL"})
        first.commit()
        second = MessageStore(db_path)
        second.close()
        assert wal_path.exists()

        # A short-lived reader (backup job, CLI, test script) opens and closes the database.
        subprocess.run([sys.executable, "-c", _OTHER_PROCESS_READER, str(db_path)], check=True)

        # It must see this process's connection and leave the WAL alone.
        assert wal_path.exists()
        first.append("lock-retention", {"role": "user", "content": "written after the reader left"})
        first.commit()
    finally:
        first.close()

    import sqlite3

    check = sqlite3.connect(db_path)
    try:
        assert check.execute("SELECT count(*) FROM messages").fetchone()[0] == 2
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        check.close()
