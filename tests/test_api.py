import os, sys, time
os.environ["GENJUTSU_TOKEN"] = "t" * 20
os.environ["GENJUTSU_DATA"] = "/tmp/gj_test_data"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from fastapi.testclient import TestClient
from PIL import Image
import app as A

class FakeImage:  # SOMENTE para testar fila e verificacao de arquivos; nunca registrado em producao
    id, kind = "fake", "image"
    def info(self): return {"id": "fake"}
    def run(self, p):
        if p["prompt"] == "erro": raise ValueError("x")
        return [Image.new("RGB", (64, 64), "red")]
A.ADAPTERS["fake"] = FakeImage()
H = {"Authorization": "Bearer " + "t" * 20}

def wait(c, jid):
    for _ in range(50):
        j = c.get(f"/jobs/{jid}", headers=H).json()
        if j["status"] in ("done", "failed", "cancelled"): return j
        time.sleep(0.1)

def test_auth_and_validation():
    with TestClient(A.app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/jobs").status_code == 401
        assert c.get("/jobs", headers={"Authorization": "Bearer errado"}).status_code == 401
        assert c.post("/jobs/image", headers=H, json={"prompt": "", "model": "fake"}).status_code == 422
        assert c.post("/jobs/image", headers=H, json={"prompt": "a", "model": "fake", "width": 500}).status_code == 422
        assert c.post("/jobs/image", headers=H, json={"prompt": "a", "model": "nao"}).status_code == 422

def test_job_done_and_file_exists():
    with TestClient(A.app) as c:
        j = c.post("/jobs/image", headers=H, json={"prompt": "a", "model": "fake"}).json()
        assert j["status"] == "queued"
        j = wait(c, j["id"])
        assert j["status"] == "done" and j["files"] == ["0.png"]
        r = c.get(f"/files/{j['id']}/0.png", headers=H)
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
        assert c.get(f"/files/{j['id']}/../x", headers=H).status_code in (404, 422)
        assert c.get(f"/files/{j['id']}/0.png").status_code == 401

def test_failure_is_reported():
    with TestClient(A.app) as c:
        j = wait(c, c.post("/jobs/image", headers=H, json={"prompt": "erro", "model": "fake"}).json()["id"])
        assert j["status"] == "failed" and "ValueError" in j["error"]

def test_idempotent_client_key():
    with TestClient(A.app) as c:
        b = {"prompt": "a", "model": "fake", "client_key": "abcdefgh1"}
        assert c.post("/jobs/image", headers=H, json=b).json()["id"] == c.post("/jobs/image", headers=H, json=b).json()["id"]

def test_cancel_only_queued():
    with TestClient(A.app) as c:
        j = wait(c, c.post("/jobs/image", headers=H, json={"prompt": "a", "model": "fake"}).json()["id"])
        assert c.post(f"/jobs/{j['id']}/cancel", headers=H).status_code == 409
