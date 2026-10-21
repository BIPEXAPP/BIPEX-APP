"""GENJUTSU AI - API e fila de tarefas. Estados reais, sem simulacao."""
import hmac, json, os, sqlite3, threading, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

DATA = Path(os.getenv("GENJUTSU_DATA", "data"))
OUT = DATA / "out"
OUT.mkdir(parents=True, exist_ok=True)
TOKEN = os.getenv("GENJUTSU_TOKEN", "")
if len(TOKEN) < 16:
    raise RuntimeError("Defina GENJUTSU_TOKEN com pelo menos 16 caracteres.")
ORIGINS = [o for o in os.getenv("GENJUTSU_ORIGINS", "https://bipexapp.github.io").split(",") if o]

db = sqlite3.connect(DATA / "jobs.db", check_same_thread=False)
db.row_factory = sqlite3.Row
lock = threading.Lock()
db.execute("""CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, client_key TEXT UNIQUE,
  kind TEXT, params TEXT, status TEXT, error TEXT, files TEXT, created REAL, updated REAL)""")


class DiffusersImage:
    """Adaptador real de texto para imagem (diffusers). Carrega o modelo so na primeira tarefa."""
    id = "image-diffusers"
    kind = "image"

    def __init__(self):
        self.model = os.getenv("GENJUTSU_IMAGE_MODEL", "stabilityai/sd-turbo")
        self.pipe = None

    def info(self):
        try:
            import diffusers, torch  # noqa
            ok, why = True, "dependencias instaladas"
        except ImportError:
            ok, why = False, "instale server/requirements-ai.txt"
        return {"id": self.id, "kind": self.kind, "model": self.model, "available": ok, "note": why,
                "license": "NAO VERIFICADA: confira a licenca oficial do modelo antes de uso comercial"}

    def run(self, p):
        import torch
        from diffusers import AutoPipelineForText2Image
        if self.pipe is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            dt = torch.float16 if dev == "cuda" else torch.float32
            self.pipe = AutoPipelineForText2Image.from_pretrained(self.model, torch_dtype=dt).to(dev)
        g = torch.Generator().manual_seed(p["seed"])
        res = self.pipe(p["prompt"], width=p["width"], height=p["height"], num_inference_steps=p["steps"],
                        guidance_scale=p["guidance"], num_images_per_prompt=p["n"], generator=g)
        return res.images


ADAPTERS = {a.id: a for a in [DiffusersImage()]}


class ImageReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    model: str = "image-diffusers"
    width: int = Field(512, ge=256, le=768, multiple_of=64)
    height: int = Field(512, ge=256, le=768, multiple_of=64)
    steps: int = Field(4, ge=1, le=50)
    guidance: float = Field(0.0, ge=0, le=15)
    n: int = Field(1, ge=1, le=4)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    client_key: Optional[str] = Field(None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


def auth(authorization: str = Header("")):
    if not hmac.compare_digest(authorization.removeprefix("Bearer "), TOKEN):
        raise HTTPException(401, "Token inválido ou ausente.")


def row(r):
    d = dict(r)
    d["params"], d["files"] = json.loads(d["params"]), json.loads(d["files"] or "[]")
    return d


def setj(jid, **kw):
    with lock:
        sets = ",".join(f"{k}=?" for k in kw)
        db.execute(f"UPDATE jobs SET {sets}, updated=? WHERE id=?", [*kw.values(), time.time(), jid])
        db.commit()


def work(jid, kind, params):
    """Executa uma tarefa e so marca 'done' se os arquivos existirem e abrirem."""
    from PIL import Image
    ad = ADAPTERS.get(params["model"])
    if ad is None or ad.kind != kind:
        return setj(jid, status="failed", error="Modelo desconhecido.")
    try:
        imgs = ad.run(params)
        names = []
        for i, im in enumerate(imgs):
            name = f"{i}.png"
            (OUT / jid).mkdir(exist_ok=True)
            im.save(OUT / jid / name)
            with Image.open(OUT / jid / name) as chk:
                chk.verify()
            names.append(name)
        if not names:
            raise RuntimeError("O modelo não retornou imagens.")
        setj(jid, status="done", files=json.dumps(names))
    except Exception as e:  # erro real, sem detalhes internos para o cliente
        print("job", jid, "falhou:", repr(e))
        setj(jid, status="failed", error=f"Falha na geração: {type(e).__name__}")


def loop(stop):
    while not stop.is_set():
        with lock:
            r = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if r:
                db.execute("UPDATE jobs SET status='processing', updated=? WHERE id=?", (time.time(), r["id"]))
                db.commit()
        if r:
            work(r["id"], r["kind"], json.loads(r["params"]))
        else:
            stop.wait(0.5)


@asynccontextmanager
async def lifespan(_):
    with lock:  # tarefas que estavam rodando quando o servidor caiu
        db.execute("UPDATE jobs SET status='failed', error='Interrompida por reinício do servidor.' WHERE status='processing'")
        db.commit()
    stop = threading.Event()
    t = threading.Thread(target=loop, args=(stop,), daemon=True)
    t.start()
    yield
    stop.set()


app = FastAPI(title="GENJUTSU AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models", dependencies=[Depends(auth)])
def models():
    return [a.info() for a in ADAPTERS.values()]


@app.post("/jobs/image", dependencies=[Depends(auth)])
def create_image(req: ImageReq):
    if req.model not in ADAPTERS or ADAPTERS[req.model].kind != "image":
        raise HTTPException(422, "Modelo de imagem desconhecido.")
    with lock:
        if req.client_key:
            old = db.execute("SELECT * FROM jobs WHERE client_key=?", (req.client_key,)).fetchone()
            if old:
                return row(old)
        jid, now = uuid.uuid4().hex, time.time()
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                   (jid, req.client_key, "image", req.model_dump_json(), "queued", None, None, now, now))
        db.commit()
        return row(db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())


@app.get("/jobs", dependencies=[Depends(auth)])
def list_jobs(limit: int = 50):
    with lock:
        rs = db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (min(max(limit, 1), 200),)).fetchall()
    return [row(r) for r in rs]


def get_job(jid):
    with lock:
        r = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if not r:
        raise HTTPException(404, "Tarefa não encontrada.")
    return r


@app.get("/jobs/{jid}", dependencies=[Depends(auth)])
def one_job(jid: str):
    return row(get_job(jid))


@app.post("/jobs/{jid}/cancel", dependencies=[Depends(auth)])
def cancel(jid: str):
    get_job(jid)
    with lock:
        n = db.execute("UPDATE jobs SET status='cancelled', updated=? WHERE id=? AND status='queued'", (time.time(), jid)).rowcount
        db.commit()
    if not n:
        raise HTTPException(409, "Só é possível cancelar tarefas que ainda estão na fila.")
    return row(get_job(jid))


@app.get("/files/{jid}/{name}", dependencies=[Depends(auth)])
def file(jid: str, name: str):
    j = row(get_job(jid))
    if name not in j["files"]:
        raise HTTPException(404, "Arquivo não encontrado.")
    p = OUT / jid / name
    if not p.is_file():
        raise HTTPException(404, "Arquivo ausente no disco.")
    return FileResponse(p, media_type="image/png")
