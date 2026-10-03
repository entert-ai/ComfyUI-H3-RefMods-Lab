from comfy_api.latest import ComfyExtension
from .nodes import NODE_CLASSES

WEB_DIRECTORY = "./web"
__version__ = "0.1.0"


class H3RefModsLabExtension(ComfyExtension):
    async def get_node_list(self):
        return NODE_CLASSES


async def comfy_entrypoint():
    from .picker_routes import register_routes
    register_routes()
    return H3RefModsLabExtension()
