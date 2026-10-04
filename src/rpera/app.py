from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path
from threading import RLock
from typing import BinaryIO

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from .agents import AGENTS, AGENT_CATALOG
from .config import CapabilitySettingsStore, DrawingPresetStore, NetworkSettingsStore, PresetStore, RuntimeSettingsStore
from .content_api import content_router
from .content_store import ContentStore
from .drawing_providers import NovelAIClient, StableDiffusionWebUIClient
from .models import AgentPresetSettingsWrite, AiFallbackSettingsWrite, CapabilitySettings, ConnectionTestResult, ContentEntityCreate, DrawingConnectionTestResult, DrawingDefaultWrite, DrawingPresetWrite, DrawingTestRequest, EventPage, ModelListRequest, NetworkSettingsWrite, PresetDuplicate, PresetWrite, RuntimeEvent, RuntimeSettings, SaveBranch, SaveCreate, SaveEntityDelete, SaveEntityMove, SaveEntityUpdate, SaveRename, TracePage, TurnCreate
from .prompt_store import PromptConfigurationError, PromptStore
from .providers import BoundModelClient, ConfiguredModelClient, ModelClient
from .runtime import AgentRuntime, EventHub, TurnAbortError
from .server_config import BasicAuthMiddleware, load_server_config
from .storage import IMAGE_ARTIFACT_MAX_BYTES, ModLibrary, SaveBusyError, SaveEntityConflictError, SaveStore, TurnAlreadyRunningError, TurnRetryError, WorldLibrary


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(
    data_dir: Path | None = None,
    model_client: ModelClient | None = None,
    prompt_dir: Path | None = None,
    mock_response_dir: Path | None = None,
    prefill_dir: Path | None = None,
) -> FastAPI:
    shared_data = data_dir
    configured_data = os.environ.get("RPERA_DATA_DIR")
    if shared_data is None and configured_data:
        shared_data = Path(configured_data)
    content_data = shared_data or PROJECT_ROOT / "data"
    user_data = shared_data or Path(os.environ.get("RPERA_USER_DATA_DIR", str(PROJECT_ROOT)))
    server_config = load_server_config(user_data)
    presets = PresetStore(user_data)
    drawing_presets = DrawingPresetStore(user_data)
    capability_settings = CapabilitySettingsStore(user_data)
    runtime_settings = RuntimeSettingsStore(user_data)
    network_settings = NetworkSettingsStore(user_data)
    content_lock = RLock()
    worlds = WorldLibrary(content_data, content_lock)
    mods = ModLibrary(content_data, content_lock)
    content = ContentStore(content_data, content_lock)
    saves = SaveStore(user_data, worlds, mods, content_lock)
    events = EventHub()
    model = model_client or ConfiguredModelClient(presets, network_settings)
    prompt_store = PromptStore(
        prompt_dir or PROJECT_ROOT / "prompts",
        AGENTS,
        mock_response_dir or PROJECT_ROOT / "mock_responses",
        prefill_dir or PROJECT_ROOT / "prefills",
    )
    runtime = AgentRuntime(
        saves,
        content,
        model,
        events,
        runtime_settings,
        prompt_store,
        capability_settings,
        drawing_presets,
        network_settings,
    )
    templates = Jinja2Templates(directory=PROJECT_ROOT / "web" / "templates")

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        await asyncio.to_thread(saves.recover_interrupted_turns)
        yield
        await runtime.shutdown()

    application = FastAPI(title="RPera", version="0.1.0", lifespan=lifespan)
    application.add_middleware(BasicAuthMiddleware, config=server_config.basic_auth)
    application.include_router(content_router(content))
    application.mount("/static", StaticFiles(directory=PROJECT_ROOT / "web" / "static"), name="static")
    application.state.presets = presets
    application.state.drawing_presets = drawing_presets
    application.state.capability_settings = capability_settings
    application.state.runtime_settings = runtime_settings
    application.state.network_settings = network_settings
    application.state.content_data_dir = content_data
    application.state.user_data_dir = user_data
    application.state.worlds = worlds
    application.state.mods = mods
    application.state.content = content
    application.state.content_lock = content_lock
    application.state.saves = saves
    application.state.events = events
    application.state.prompt_store = prompt_store
    application.state.runtime = runtime

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        static_version = max(
            (PROJECT_ROOT / "web" / "static" / "app.css").stat().st_mtime_ns,
            (PROJECT_ROOT / "web" / "static" / "app.js").stat().st_mtime_ns,
            (PROJECT_ROOT / "web" / "static" / "trace-markdown.js").stat().st_mtime_ns,
        )
        response = templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"static_version": static_version},
        )
        response.headers["Cache-Control"] = "no-cache"
        return response

    @application.get("/api/worlds")
    def list_worlds():
        try:
            return worlds.list_worlds()
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/mods")
    def list_mods(page: int = Query(1, ge=1)):
        try:
            return mods.list_mods(page=page)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/worlds/{world_name}/scenarios")
    def list_scenarios(world_name: str, mod_name: list[str] = Query(default=[])):
        try:
            return saves.scenario_options(world_name, mod_name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/saves")
    def list_saves(page: int = Query(1, ge=1)):
        return saves.list_saves(page=page)

    @application.post("/api/saves", status_code=201)
    def create_save(payload: SaveCreate):
        try:
            return saves.create_save(payload)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/saves/{save_id}")
    def get_save(save_id: str):
        try:
            return saves.get_save(save_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.post("/api/saves/{save_id}/rename")
    def rename_save(save_id: str, payload: SaveRename):
        try:
            return saves.rename_save(save_id, payload.name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except SaveBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.delete("/api/saves/{save_id}")
    async def delete_save(save_id: str):
        try:
            await runtime.delete_save(save_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except SaveBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"id": save_id, "deleted": True}

    @application.get("/api/saves/{save_id}/entities")
    def list_entities(save_id: str):
        try:
            return saves.list_entities(save_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.post("/api/saves/{save_id}/entities", status_code=201)
    def create_save_entity(save_id: str, payload: ContentEntityCreate):
        try:
            return saves.create_save_entity(save_id, payload.name, payload.type)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (SaveBusyError, SaveEntityConflictError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/saves/{save_id}/entity")
    def get_save_entity(save_id: str, path: str):
        try:
            return saves.get_save_entity(save_id, path)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.put("/api/saves/{save_id}/entity")
    def update_save_entity(save_id: str, path: str, payload: SaveEntityUpdate):
        try:
            return saves.update_save_entity(save_id, path, payload)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (SaveBusyError, SaveEntityConflictError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.post("/api/saves/{save_id}/entity/rename")
    def move_save_entity(save_id: str, path: str, payload: SaveEntityMove):
        try:
            return saves.move_save_entity(save_id, path, payload.name, payload.type, payload.revision)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (SaveBusyError, SaveEntityConflictError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.delete("/api/saves/{save_id}/entity")
    def delete_save_entity(save_id: str, path: str, payload: SaveEntityDelete):
        try:
            return saves.delete_save_entity(save_id, path, payload.revision)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (SaveBusyError, SaveEntityConflictError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/saves/{save_id}/turns")
    def list_turns(save_id: str):
        try:
            return saves.list_turns(save_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get("/api/saves/{save_id}/turns/{turn_id}")
    def get_turn(save_id: str, turn_id: str):
        try:
            return saves.get_turn(save_id, turn_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.post("/api/saves/{save_id}/turns/{turn_id}/branch", status_code=201)
    async def branch_save(save_id: str, turn_id: str, payload: SaveBranch):
        try:
            return await runtime.branch_save(save_id, turn_id, payload.name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post("/api/saves/{save_id}/turns", status_code=202)
    async def create_turn(save_id: str, request: Request):
        async def start(player_input: str, images: list[UploadFile]):
            uploads: list[tuple[str, BinaryIO]] = []
            for image in images:
                if image.size is not None and image.size > IMAGE_ARTIFACT_MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"图片超过 {IMAGE_ARTIFACT_MAX_BYTES // (1024 * 1024)} MiB 上限：{image.filename or 'image'}",
                    )
                uploads.append((image.filename or "image", image.file))
            try:
                turn_id = await runtime.start_turn(save_id, player_input, uploads)
            except TurnAlreadyRunningError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            except PromptConfigurationError as error:
                raise HTTPException(status_code=500, detail=str(error)) from error
            except KeyError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            turn = saves.get_turn(save_id, turn_id)
            return {"turn_id": turn_id, "status": "running", "images": turn.images}

        content_type = request.headers.get("content-type", "").lower()
        if content_type.startswith("application/json"):
            try:
                player_input = TurnCreate.model_validate(await request.json()).content
            except (ValueError, TypeError) as error:
                raise HTTPException(status_code=422, detail="玩家输入不是有效的回合请求") from error
            return await start(player_input, [])
        elif content_type.startswith("multipart/form-data"):
            parser = MultiPartParser(request.headers, request.stream(), max_files=float("inf"))
            parser.spool_max_size = 1
            try:
                form = await parser.parse()
            except MultiPartException as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            try:
                raw_content = form.get("content", "")
                if not isinstance(raw_content, str):
                    raise HTTPException(status_code=422, detail="玩家输入必须是文本")
                player_input = raw_content.strip()
                if len(player_input) > 20_000:
                    raise HTTPException(status_code=422, detail="玩家输入不能超过 20000 个字符")
                image_fields = form.getlist("images")
                images = [item for item in image_fields if isinstance(item, UploadFile)]
                if len(images) != len(image_fields):
                    raise HTTPException(status_code=422, detail="图片字段必须是上传文件")
                if not player_input and not images:
                    raise HTTPException(status_code=422, detail="玩家输入和图片不能同时为空")
                return await start(player_input, images)
            finally:
                await form.close()
        raise HTTPException(status_code=415, detail="回合请求只支持 JSON 或 multipart/form-data")

    @application.get("/api/saves/{save_id}/turns/{turn_id}/images/{image_id}/content")
    def get_turn_image(save_id: str, turn_id: str, image_id: str):
        try:
            path, mime_type, _ = saves.get_turn_image_content(save_id, turn_id, image_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return FileResponse(
            path,
            media_type=mime_type,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.get("/api/saves/{save_id}/turns/{turn_id}/generated-images/{part_id}/content")
    def get_generated_image(save_id: str, turn_id: str, part_id: str):
        try:
            path, mime_type = saves.get_generated_image_content(save_id, turn_id, part_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return FileResponse(
            path,
            media_type=mime_type,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.post("/api/saves/{save_id}/turns/{turn_id}/retry", status_code=202)
    async def retry_turn(save_id: str, turn_id: str):
        try:
            retried_turn_id = await runtime.retry_turn(save_id, turn_id)
        except TurnRetryError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except TurnAlreadyRunningError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except PromptConfigurationError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"turn_id": retried_turn_id, "status": "running", "resumed": True}

    @application.post("/api/saves/{save_id}/turns/{turn_id}/abort")
    async def abort_turn(save_id: str, turn_id: str):
        try:
            aborted_turn_id = await runtime.abort_turn(save_id, turn_id)
        except TurnAbortError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"turn_id": aborted_turn_id, "status": "interrupted"}

    @application.get("/api/saves/{save_id}/events/history", response_model=EventPage)
    async def list_event_history(save_id: str, page: int | None = Query(default=None, ge=1)) -> EventPage:
        try:
            await asyncio.to_thread(saves.get_save, save_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return await asyncio.to_thread(saves.list_event_page, save_id, page)

    @application.get("/api/saves/{save_id}/trace/history", response_model=TracePage)
    async def list_trace_history(save_id: str, page: int | None = Query(default=None, ge=1)) -> TracePage:
        try:
            await asyncio.to_thread(saves.get_save, save_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return await asyncio.to_thread(saves.list_trace_page, save_id, page)

    @application.get("/api/saves/{save_id}/events")
    async def stream_events(
        request: Request,
        save_id: str,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        try:
            await asyncio.to_thread(saves.get_save, save_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        last_event_id = request.headers.get("last-event-id")
        if last_event_id and last_event_id.isdigit():
            after = max(after, int(last_event_id))
        try:
            queue = events.subscribe(save_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        async def generate() -> AsyncIterator[str]:
            last_id = after
            try:
                while True:
                    existing = await asyncio.to_thread(saves.list_events, save_id, last_id, 500)
                    for event in existing:
                        last_id = event.id
                        yield format_sse(event)
                    if len(existing) < 500:
                        break
                while True:
                    try:
                        close_stream = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    if close_stream:
                        return
                    while True:
                        pending = await asyncio.to_thread(saves.list_events, save_id, last_id, 500)
                        for event in pending:
                            last_id = event.id
                            yield format_sse(event)
                        if len(pending) < 500:
                            break
            finally:
                events.unsubscribe(save_id, queue)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @application.get("/api/ai-presets")
    def list_presets():
        return presets.public()

    @application.get("/api/drawing-presets")
    def list_drawing_presets():
        return drawing_presets.public()

    @application.get("/api/capability-settings")
    def get_capability_settings():
        return capability_settings.get()

    @application.put("/api/capability-settings")
    def update_capability_settings(payload: CapabilitySettings):
        return capability_settings.update(payload)

    @application.post("/api/drawing-presets", status_code=201)
    def create_drawing_preset(payload: DrawingPresetWrite):
        try:
            return drawing_presets.create(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.put("/api/drawing-presets/{preset_id}")
    def update_drawing_preset(preset_id: str, payload: DrawingPresetWrite):
        try:
            return drawing_presets.update(preset_id, payload)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.delete("/api/drawing-presets/{preset_id}")
    def delete_drawing_preset(preset_id: str):
        try:
            return drawing_presets.delete(preset_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.post("/api/drawing-presets/{preset_id}/duplicate")
    def duplicate_drawing_preset(preset_id: str, payload: PresetDuplicate):
        try:
            return drawing_presets.duplicate(preset_id, payload.name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.put("/api/drawing-preset-settings")
    def set_default_drawing_preset(payload: DrawingDefaultWrite):
        try:
            return drawing_presets.set_default(payload.preset_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get("/api/drawing-presets/{preset_id}/resources")
    async def drawing_resources(preset_id: str):
        try:
            preset = await asyncio.to_thread(drawing_presets.get, preset_id)
            if preset.provider != "stable_diffusion_webui":
                raise HTTPException(status_code=400, detail="NovelAI 不提供 WebUI 资源列表")
            client = StableDiffusionWebUIClient(
                preset,
                network_settings=await asyncio.to_thread(network_settings.get),
            )
            return await client.resources()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @application.post("/api/drawing-presets/{preset_id}/test")
    async def test_drawing_preset(preset_id: str):
        try:
            preset = await asyncio.to_thread(drawing_presets.get, preset_id)
            client = (NovelAIClient if preset.provider == "novelai" else StableDiffusionWebUIClient)(
                preset,
                network_settings=await asyncio.to_thread(network_settings.get),
            )
            started = time.monotonic()
            active_checkpoint = await client.test_connection()
            return DrawingConnectionTestResult(ok=True, latency_ms=round((time.monotonic() - started) * 1000), active_checkpoint=active_checkpoint)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @application.post("/api/drawing-presets/{preset_id}/test-generation")
    async def generate_drawing_test(preset_id: str, payload: DrawingTestRequest):
        try:
            preset = await asyncio.to_thread(drawing_presets.get, preset_id)
            client = (NovelAIClient if preset.provider == "novelai" else StableDiffusionWebUIClient)(
                preset,
                network_settings=await asyncio.to_thread(network_settings.get),
            )
            settings = await asyncio.to_thread(capability_settings.get)
            return await client.generate_test(
                payload.content_prompt,
                settings.character_portrait_generation,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @application.get("/api/agent-preset-settings")
    def get_agent_preset_settings():
        collection = presets.public()
        return {
            "main_preset_id": collection.main_preset_id,
            "agent_preset_overrides": collection.agent_preset_overrides,
            "agent_streaming": collection.agent_streaming,
            "agents": [agent.__dict__ for agent in AGENT_CATALOG],
        }

    @application.get("/api/ai-fallback-settings")
    def get_ai_fallback_settings():
        collection = presets.public()
        return {"enabled": collection.fallback_enabled, "preset_id": collection.fallback_preset_id}

    @application.put("/api/ai-fallback-settings")
    def update_ai_fallback_settings(payload: AiFallbackSettingsWrite):
        try:
            collection = presets.update_fallback_settings(payload)
            return {"enabled": collection.fallback_enabled, "preset_id": collection.fallback_preset_id}
        except KeyError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.put("/api/agent-preset-settings")
    def update_agent_preset_settings(payload: AgentPresetSettingsWrite):
        try:
            return presets.update_agent_settings(payload)
        except KeyError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get("/api/runtime-settings")
    def get_runtime_settings():
        return runtime_settings.get()

    @application.put("/api/runtime-settings")
    def update_runtime_settings(payload: RuntimeSettings):
        try:
            return runtime_settings.update(payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get("/api/network-settings")
    def get_network_settings():
        return network_settings.public()

    @application.put("/api/network-settings")
    def update_network_settings(payload: NetworkSettingsWrite):
        try:
            return network_settings.update(payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/ai-presets", status_code=201)
    def create_preset(payload: PresetWrite):
        try:
            return presets.create(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.put("/api/ai-presets/{preset_id}")
    def update_preset(preset_id: str, payload: PresetWrite):
        try:
            return presets.update(preset_id, payload)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.delete("/api/ai-presets/{preset_id}")
    def delete_preset(preset_id: str):
        try:
            return presets.delete(preset_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post("/api/ai-presets/{preset_id}/duplicate")
    def duplicate_preset(preset_id: str, payload: PresetDuplicate):
        try:
            return presets.duplicate(preset_id, payload.name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.post("/api/ai-presets/models")
    async def list_preset_models(payload: ModelListRequest):
        try:
            client = BoundModelClient(
                await asyncio.to_thread(presets.model_list_preset, payload),
                network_settings=await asyncio.to_thread(network_settings.get),
            )
            return await client.list_models()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @application.post("/api/ai-presets/{preset_id}/test")
    async def test_preset(preset_id: str):
        try:
            client = BoundModelClient(
                await asyncio.to_thread(presets.get, preset_id),
                network_settings=await asyncio.to_thread(network_settings.get),
            )
            started = time.monotonic()
            await client.test_connection()
            return ConnectionTestResult(
                ok=True,
                latency_ms=round((time.monotonic() - started) * 1000),
                model=client.preset.model,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return application


def format_sse(event: RuntimeEvent) -> str:
    data = event.model_dump_json()
    return f"id: {event.id}\nevent: runtime\ndata: {data}\n\n"


app = create_app()
