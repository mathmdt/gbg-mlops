"""ingest.py uploads each raw file once per content, with its SHA-256 as metadata."""

import hashlib

from botocore.exceptions import ClientError

from pipelines.ingest import BUCKET, RAW_FILES, ingest


class FakeS3:
    def __init__(self):
        self.objects, self.uploads = {}, []

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"Metadata": self.objects[Bucket, Key]}

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.objects[bucket, key] = ExtraArgs["Metadata"]
        self.uploads.append(key)


def test_uploads_with_sha256_and_skips_unchanged_files(tmp_path):
    for name in RAW_FILES:
        (tmp_path / name).write_bytes(name.encode())
    s3 = FakeS3()

    digests = ingest(tmp_path, s3)
    assert s3.uploads == list(RAW_FILES)
    for name in RAW_FILES:
        assert digests[name] == hashlib.sha256(name.encode()).hexdigest()
        assert s3.objects[BUCKET, name] == {"sha256": digests[name]}

    ingest(tmp_path, s3)
    assert len(s3.uploads) == 3  # nothing changed: nothing sent

    (tmp_path / RAW_FILES[0]).write_bytes(b"new version")
    ingest(tmp_path, s3)
    assert s3.uploads[3:] == [RAW_FILES[0]]
