"""Preview only explicitly selected Lab packs; no VAE or text-encoder work."""
import asyncio

from aiohttp import web
import folder_paths

from . import core
from .nodes import FOLDER


def resolve_plan(plan, depth=0):
    if depth > 16 or not isinstance(plan, dict):
        raise ValueError("Source preview graph is too deep or invalid.")
    kind = plan.get("type")
    if kind == "load":
        filename = plan.get("filename")
        if not isinstance(filename, str) or filename not in folder_paths.get_filename_list(FOLDER):
            raise ValueError("Choose an existing Lab RefMod in the connected Load node.")
        return core.load_pack(folder_paths.get_full_path_or_raise(FOLDER, filename))
    if kind == "combine":
        if "inputs" in plan:
            inputs = plan["inputs"]
            if not isinstance(inputs, list) or len(inputs) > 100:
                raise ValueError("Combine preview accepts up to 100 inputs.")
            return core.combine_many([(resolve_plan(item["input"], depth + 1), item.get("strength", 1)) for item in inputs])
        return core.combine(resolve_plan(plan["a"], depth + 1), resolve_plan(plan["b"], depth + 1),
                            float(plan["strength_a"]), float(plan["strength_b"]))
    if kind == "instructions":
        fields = {name: plan[name] for name in ("description", "subject_name", "retention_strategy", "retention_details", "audio_retention_strategy", "audio_retention_details") if name in plan}
        return core.set_instructions(resolve_plan(plan["input"], depth + 1), **fields)
    if kind == "select":
        return core.select_sources(resolve_plan(plan["input"], depth + 1), plan["selection"])
    raise ValueError("Refresh sources supports connected Load, Combine and Select Sources nodes. Save an in-memory pack first, or queue this selector alone to preview it.")


async def preview_sources(request):
    try:
        plan = await request.json()
        def preview():
            return core.source_catalog(resolve_plan(plan))
        sources = await asyncio.to_thread(preview)
        return web.json_response({"sources": sources})
    except (ValueError, KeyError, TypeError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


def register_routes():
    from server import PromptServer
    PromptServer.instance.routes.post("/h3-refmods-lab/sources")(preview_sources)
