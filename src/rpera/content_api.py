from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException

from .content_store import (
    ContentConflictError,
    ContentNotFoundError,
    ContentStore,
    SourceKind,
    UnsafeContentError,
)
from .models import (
    ContentEntityCreate,
    ContentEntityMove,
    ContentEntityUpdate,
    ContentRename,
    ContentScenarioCreate,
    ContentScenarioUpdate,
    ContentSourceCreate,
    ContentSourceUpdate,
    ContentStyleCreate,
    ContentStyleEnabledUpdate,
    ContentStyleUpdate,
)


def _execute(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except ContentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ContentConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except UnsafeContentError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def content_router(store: ContentStore) -> APIRouter:
    router = APIRouter()
    router.include_router(_source_router(store, "world", "worlds"))
    router.include_router(_source_router(store, "mod", "mods"))
    router.include_router(_style_router(store))
    return router


def _style_router(store: ContentStore) -> APIRouter:
    router = APIRouter(prefix="/api/content/styles", tags=["content"])

    @router.get("")
    def list_styles():
        return _execute(store.list_styles)

    @router.post("", status_code=201)
    def create_style(payload: ContentStyleCreate):
        return _execute(lambda: store.create_style(payload))

    @router.get("/{style_name}")
    def get_style(style_name: str):
        return _execute(lambda: store.get_style(style_name))

    @router.put("/{style_name}")
    def update_style(style_name: str, payload: ContentStyleUpdate):
        return _execute(lambda: store.update_style(style_name, payload))

    @router.put("/{style_name}/enabled")
    def set_style_enabled(style_name: str, payload: ContentStyleEnabledUpdate):
        return _execute(lambda: store.set_style_enabled(style_name, payload.enabled))

    @router.post("/{style_name}/rename")
    def rename_style(style_name: str, payload: ContentRename):
        return _execute(lambda: store.rename_style(style_name, payload))

    @router.delete("/{style_name}")
    def delete_style(style_name: str):
        return _execute(lambda: store.delete_style(style_name))

    return router


def _source_router(store: ContentStore, kind: SourceKind, plural: str) -> APIRouter:
    router = APIRouter(prefix=f"/api/content/{plural}", tags=["content"])

    @router.get("")
    def list_sources():
        return _execute(lambda: store.list_sources(kind))

    @router.post("", status_code=201)
    def create_source(payload: ContentSourceCreate):
        return _execute(lambda: store.create_source(kind, payload))

    @router.get("/{source_name}")
    def get_source(source_name: str):
        return _execute(lambda: store.get_source(kind, source_name))

    @router.put("/{source_name}")
    def update_source(source_name: str, payload: ContentSourceUpdate):
        return _execute(lambda: store.update_source(kind, source_name, payload))

    @router.post("/{source_name}/rename")
    def rename_source(source_name: str, payload: ContentRename):
        return _execute(lambda: store.rename_source(kind, source_name, payload))

    @router.delete("/{source_name}")
    def delete_source(source_name: str):
        return _execute(lambda: store.delete_source(kind, source_name))

    @router.get("/{source_name}/scenarios")
    def list_scenarios(source_name: str):
        return _execute(lambda: store.list_scenarios(kind, source_name))

    @router.post("/{source_name}/scenarios", status_code=201)
    def create_scenario(source_name: str, payload: ContentScenarioCreate):
        return _execute(lambda: store.create_scenario(kind, source_name, payload))

    @router.put("/{source_name}/scenarios/{scenario_name}")
    def update_scenario(source_name: str, scenario_name: str, payload: ContentScenarioUpdate):
        return _execute(lambda: store.update_scenario(kind, source_name, scenario_name, payload))

    @router.post("/{source_name}/scenarios/{scenario_name}/rename")
    def rename_scenario(source_name: str, scenario_name: str, payload: ContentRename):
        return _execute(lambda: store.rename_scenario(kind, source_name, scenario_name, payload))

    @router.delete("/{source_name}/scenarios/{scenario_name}")
    def delete_scenario(source_name: str, scenario_name: str):
        return _execute(lambda: store.delete_scenario(kind, source_name, scenario_name))

    @router.get("/{source_name}/entities")
    def list_entities(source_name: str):
        return _execute(lambda: store.list_entities(kind, source_name))

    @router.post("/{source_name}/entities", status_code=201)
    def create_entity(source_name: str, payload: ContentEntityCreate):
        return _execute(lambda: store.create_entity(kind, source_name, payload))

    @router.get("/{source_name}/entities/{entity_name}")
    def get_entity(source_name: str, entity_name: str):
        return _execute(lambda: store.get_entity(kind, source_name, entity_name))

    @router.put("/{source_name}/entities/{entity_name}")
    def update_entity(source_name: str, entity_name: str, payload: ContentEntityUpdate):
        return _execute(lambda: store.update_entity(kind, source_name, entity_name, payload))

    @router.post("/{source_name}/entities/{entity_name}/rename")
    def move_entity(source_name: str, entity_name: str, payload: ContentEntityMove):
        return _execute(lambda: store.move_entity(kind, source_name, entity_name, payload))

    @router.delete("/{source_name}/entities/{entity_name}")
    def delete_entity(source_name: str, entity_name: str):
        return _execute(lambda: store.delete_entity(kind, source_name, entity_name))

    return router
