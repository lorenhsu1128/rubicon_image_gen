"""Minimal ComfyUI HTTP client: upload images, queue a prompt, wait for it, download outputs."""
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, url: str, timeout_s: float = 3600):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.client_id = uuid.uuid4().hex

    def _request(self, path: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ComfyError(f"{path}: HTTP {e.code} {e.read().decode(errors='replace')}") from e

    def _json(self, path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        body = self._request(path, data, {"Content-Type": "application/json"} if data else None)
        return json.loads(body or b"null")

    def system_stats(self) -> dict:
        return self._json("/system_stats")

    def upload(self, path: Path) -> str:
        """Upload an image to ComfyUI's input dir under a content-addressed name; return that name."""
        content = path.read_bytes()
        name = f"{hashlib.sha256(content).hexdigest()[:12]}_{path.name}"
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n".encode()
            + content
            + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue"
              f"\r\n--{boundary}--\r\n".encode()
        )
        res = json.loads(self._request("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        return res["name"] if not res.get("subfolder") else f"{res['subfolder']}/{res['name']}"

    def queue(self, workflow: dict) -> str:
        res = self._json("/prompt", {"prompt": workflow, "client_id": self.client_id})
        if res.get("node_errors"):
            raise ComfyError(f"node errors: {json.dumps(res['node_errors'])}")
        return res["prompt_id"]

    def wait(self, prompt_id: str, poll_s: float = 1.0) -> dict:
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            history = self._json(f"/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry["status"]
                if status["status_str"] != "success":
                    raise ComfyError(f"prompt {prompt_id} failed: {json.dumps(status['messages'])}")
                return entry
            time.sleep(poll_s)
        raise ComfyError(f"prompt {prompt_id} timed out after {self.timeout_s}s")

    def output_images(self, entry: dict) -> list[bytes]:
        images = []
        for node_out in entry["outputs"].values():
            for img in node_out.get("images", []):
                if img.get("type") != "output":
                    continue
                query = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
                images.append(self._request(f"/view?{query}"))
        return images
