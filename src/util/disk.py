import gzip

from diskcache import FanoutCache, Disk
from diskcache.core import MODE_BINARY
from io import BytesIO

from util.logconf import logging
import os

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class GzipDisk(Disk):
    def store(self, value, read, key=None):
        """
        Override from base class diskcache.Disk.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large buffers
          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and
          compression and decompression operations did not properly handle results of
          2 or 4 GiB.

        :param value: value to convert
        :param bool read: True when value is file-like object
        :return: (size, mode, filename, value) tuple for Cache table
        """
        # pylint: disable=unidiomatic-typecheck
        if type(value) is bytes:
            if read:
                value = value.read()
                read = False

            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode="wb", compresslevel=1, fileobj=str_io)

            for offset in range(0, len(value), 2**30):
                gz_file.write(value[offset : offset + 2**30])
            gz_file.close()

            value = str_io.getvalue()

        return super(GzipDisk, self).store(value, read)

    def fetch(self, mode, filename, value, read):
        """
        Override from base class diskcache.Disk.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large buffers
          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and
          compression and decompression operations did not properly handle results of
          2 or 4 GiB.

        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        """
        value = super(GzipDisk, self).fetch(mode, filename, value, read)

        if mode == MODE_BINARY:
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode="rb", fileobj=str_io)
            read_csio = BytesIO()

            while True:
                uncompressed_data = gz_file.read(2**30)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break

            value = read_csio.getvalue()

        return value


def getCache(scope_str, base_dir=None):
    # To customize cache location, set env AUTO_QC_CACHE_DIR to an absolute path
    # e.g., export AUTO_QC_CACHE_DIR=/mnt/fast/cache/automated-qc

    if base_dir is None:
        base_dir = os.environ.get("AUTO_QC_CACHE_DIR")
        if not base_dir:
            xdg = os.environ.get("XDG_CACHE_HOME")
            if xdg:
                base_dir = os.path.join(xdg, "auto-qc", "cache")
            else:
                win = os.environ.get("LOCALAPPDATA")
                if win:
                    base_dir = os.path.join(win, "auto-qc", "cache")
                else:
                    base_dir = os.path.join("data-unversioned", "cache")

    base_dir = os.path.abspath(os.path.expanduser(base_dir))
    cache_path = os.path.join(base_dir, scope_str)

    return FanoutCache(
        cache_path,
        disk=GzipDisk,
        shards=64,
        timeout=1,
        size_limit=3e11,
        # disk_min_file_size=2**20,
    )
