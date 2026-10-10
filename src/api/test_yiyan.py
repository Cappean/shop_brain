from contextlib import asynccontextmanager

from anyio import Path
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.responses import FileResponse
from sklearn import base
from pathlib import Path
HITOKOTO_URL = "https://v1.hitokoto.cn/"
TIMEOUT = 5.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时创建复用的异步客户端，关闭时释放
    app.state.client = httpx.AsyncClient(timeout=TIMEOUT)
    yield
    await app.state.client.aclose()


app = FastAPI(
    title="Hitokoto API",
    description="基于一言网 v1 接口的简单后端",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", summary="服务说明")
async def root():
    return {
        "service": "Hitokoto API",
        "endpoints": {
            "/hitokoto": "返回 JSON 格式的句子信息",
            "/hitokoto/text": "返回纯文本句子",
        },
    }


@app.get("/hitokoto", summary="获取一言（JSON）")
async def get_hitokoto(
    c: list[str] | None = Query(
        None,
        description="句子类型，可重复传参组合，如 ?c=a&c=i。a=动画 b=漫画 c=游戏 d=文学 e=原创 f=网络 g=其他 h=影视 i=诗词 j=网易云 k=哲学 l=抖机灵",
    ),
    min_length: int | None = Query(None, ge=1, le=100, description="句子最小长度"),
    max_length: int | None = Query(None, ge=1, le=100, description="句子最大长度"),
):
    params: dict = {"encode": "json"}
    if c:
        params["c"] = c  # httpx 会把 list 展开成多个同名参数
    if min_length is not None:
        params["min_length"] = min_length
    if max_length is not None:
        params["max_length"] = max_length

    try:
        resp = await app.state.client.get(HITOKOTO_URL, params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"一言接口返回错误状态: {e.response.status_code}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"请求一言接口失败: {e}")

    return resp.json()

@app.get("/index.html", response_class=FileResponse,summary="获取一言（HTML)")
def get_hitokoto_html():
    base = Path(__file__).resolve().parent
    return FileResponse(base / "index.html")


@app.get("/hitokoto/text", response_class=PlainTextResponse, summary="获取一言（纯文本）")
async def get_hitokoto_text(
    c: str | None = Query(None, description="句子类型代码，单值"),
):
    params = {"encode": "text"}
    if c:
        params["c"] = c

    try:
        resp = await app.state.client.get(HITOKOTO_URL, params=params)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"请求一言接口失败: {e}")

    return resp.text


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)