import logging
import threading
from spandrel import ModelLoader, ImageModelDescriptor
from comfy import model_management
import torch
import comfy.utils
import folder_paths

# Try to import extension API; if not available, we still support legacy loading.
try:
    from typing_extensions import override
    from comfy_api.latest import ComfyExtension, io
    HAVE_EXTENSION_API = True
except Exception:
    HAVE_EXTENSION_API = False


# ========== Optional: spandrel extra arches ==========

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logging.info("multiGPU_upscaler: spandrel_extra_arches loaded (extra upscale architectures).")
except Exception:
    pass


# ========== Utilities ==========

def get_cuda_devices_sorted_free():
    """
    Returns list of (index, free_bytes, total_bytes) for all CUDA devices,
    sorted by free memory (descending).
    """
    if not torch.cuda.is_available():
        return []
    devs = []
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            free, total = (0, 0)
        devs.append((i, free, total))
    devs.sort(key=lambda x: x[1], reverse=True)
    return devs


def parse_device_list(selection: str, max_gpus: int) -> list[int]:
    """
    Parse device_list string:
    - 'auto' -> []
    - 'cuda:0,cuda:1' or '0,1,2' -> [0,1,2]
    Filters to valid indices < max_gpus.
    """
    if not selection:
        return []
    s = selection.strip().lower()
    if s == "auto":
        return []
    parts = [p.strip() for p in selection.split(",") if p.strip()]
    indices = []
    for p in parts:
        if p.lower().startswith("cuda:"):
            p = p.split("cuda:")[1]
        try:
            idx = int(p)
        except ValueError:
            continue
        if 0 <= idx < max_gpus and idx not in indices:
            indices.append(idx)
    return indices


def run_tiled_on_device(
    model_descriptor,
    images_bchw,
    device,
    tag,
    start_tile=512,
    min_tile=128,
    overlap=32,
):
    """
    Run comfy.utils.tiled_scale on [B, C, H, W] using a dedicated model instance on one device.
    Returns [B, H, W, C] on CPU.
    """
    scale = float(model_descriptor.scale)
    base_model = model_descriptor.model

    # Independent model instance per device when possible
    if hasattr(base_model, "state_dict") and hasattr(base_model, "load_state_dict"):
        state = base_model.state_dict()
        m = type(base_model)().to(device)
        m.load_state_dict(state)
    else:
        # Fallback: share instance (works for eval in most cases)
        m = base_model.to(device)

    m.eval()
    x = images_bchw.to(device)
    tile = start_tile

    while True:
        try:
            steps = x.shape[0] * comfy.utils.get_tiled_scale_steps(
                x.shape[3],
                x.shape[2],
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
            )
            pbar = comfy.utils.ProgressBar(steps)
            logging.info(
                f"[{tag}] start on {device}, B={x.shape[0]}, tile={tile}, "
                f"overlap={overlap}, steps={steps}"
            )

            out = comfy.utils.tiled_scale(
                x,
                lambda t: m(t),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=scale,
                pbar=pbar,
            )

            logging.info(f"[{tag}] done on {device} with tile={tile}.")
            break

        except (torch.cuda.OutOfMemoryError, model_management.OOM_EXCEPTION):
            torch.cuda.empty_cache()
            tile //= 2
            logging.warning(f"[{tag}] OOM on {device}, retry with tile={tile}")
            if tile < min_tile:
                logging.error(f"[{tag}] OOM even at minimum tile on {device}")
                raise

    # Cleanup
    try:
        m.to("cpu")
    except Exception:
        pass
    torch.cuda.empty_cache()

    out = torch.clamp(out.movedim(-3, -1), min=0.0, max=1.0)
    return out.cpu()


# ========== Node Implementations (extension API if available) ==========

if HAVE_EXTENSION_API:
    # Using comfy_api.latest.io style

    class UpscaleModelLoader(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="multiGPU_UpscaleModelLoader",
                display_name="multiGPU_upscaler: Load Upscale Model",
                category="multiGPU_upscaler",
                inputs=[
                    io.Combo.Input(
                        "model_name",
                        options=folder_paths.get_filename_list("upscale_models"),
                    ),
                ],
                outputs=[io.UpscaleModel.Output()],
            )

        @classmethod
        def execute(cls, model_name) -> io.NodeOutput:
            model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
                sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
            out = ModelLoader().load_from_state_dict(sd).eval()
            if not isinstance(out, ImageModelDescriptor):
                raise Exception("Upscale model must be a single-image model.")
            return io.NodeOutput(out)


    class ImageUpscaleWithModel(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="multiGPU_ImageUpscaleWithModel",
                display_name="multiGPU_upscaler: Single GPU Tiled",
                category="multiGPU_upscaler",
                inputs=[
                    io.UpscaleModel.Input("upscale_model"),
                    io.Image.Input("image"),
                    io.Int.Input("tile_size", default=512, min=64, max=2048, step=64),
                    io.Int.Input("min_tile_size", default=128, min=32, max=1024, step=32),
                    io.Int.Input("overlap", default=32, min=0, max=256, step=4),
                ],
                outputs=[io.Image.Output()],
            )

        @classmethod
        def execute(cls, upscale_model, image, tile_size, min_tile_size, overlap) -> io.NodeOutput:
            device = model_management.get_torch_device()
            in_img = image.movedim(-1, -3)
            if in_img.ndim == 3:
                in_img = in_img.unsqueeze(0)

            out = run_tiled_on_device(
                upscale_model,
                in_img,
                device,
                tag="SingleGPU",
                start_tile=tile_size,
                min_tile=min_tile_size,
                overlap=overlap,
            )
            return io.NodeOutput(out)


    class ImageUpscaleWithModelMultiGPU(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="multiGPU_ImageUpscaleWithModelMultiGPU",
                display_name="multiGPU_upscaler: Multi-GPU Batch Parallel",
                category="multiGPU_upscaler",
                inputs=[
                    io.UpscaleModel.Input("upscale_model"),
                    io.Image.Input("image"),

                    io.String.Input("device_list", default="auto"),
                    io.Int.Input("auto_max_devices", default=2, min=1, max=10, step=1),
                    io.Float.Input("primary_share", default=0.5, min=0.1, max=0.9, step=0.05),

                    io.Int.Input("tile_size", default=512, min=64, max=2048, step=64),
                    io.Int.Input("min_tile_size", default=128, min=32, max=1024, step=32),
                    io.Int.Input("overlap", default=32, min=0, max=256, step=4),
                ],
                outputs=[io.Image.Output()],
            )

        @classmethod
        def execute(
            cls,
            upscale_model,
            image,
            device_list,
            auto_max_devices,
            primary_share,
            tile_size,
            min_tile_size,
            overlap,
        ) -> io.NodeOutput:
            return io.NodeOutput(
                _multi_gpu_execute(
                    upscale_model,
                    image,
                    device_list,
                    auto_max_devices,
                    primary_share,
                    tile_size,
                    min_tile_size,
                    overlap,
                )
            )


    class UpscaleModelExtension(ComfyExtension):
        @override
        async def get_node_list(self):
            return [
                UpscaleModelLoader,
                ImageUpscaleWithModel,
                ImageUpscaleWithModelMultiGPU,
            ]


    async def comfy_entrypoint():
        return UpscaleModelExtension()

else:
    # Legacy style (no comfy_api.latest):
    # We'll define simple node classes compatible with NODE_CLASS_MAPPINGS.

    class UpscaleModelLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "model_name": (
                        folder_paths.get_filename_list("upscale_models"),
                        {},
                    ),
                }
            }

        RETURN_TYPES = ("UPSCALE_MODEL",)
        FUNCTION = "load"
        CATEGORY = "multiGPU_upscaler"

        def load(self, model_name):
            model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
                sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
            out = ModelLoader().load_from_state_dict(sd).eval()
            if not isinstance(out, ImageModelDescriptor):
                raise Exception("Upscale model must be a single-image model.")
            return (out,)


    class ImageUpscaleWithModel:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "upscale_model": ("UPSCALE_MODEL",),
                    "image": ("IMAGE",),
                    "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                    "min_tile_size": ("INT", {"default": 128, "min": 32, "max": 1024, "step": 32}),
                    "overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 4}),
                }
            }

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "upscale"
        CATEGORY = "multiGPU_upscaler"

        def upscale(self, upscale_model, image, tile_size, min_tile_size, overlap):
            device = model_management.get_torch_device()
            in_img = image.movedim(-1, -3)
            if in_img.ndim == 3:
                in_img = in_img.unsqueeze(0)
            out = run_tiled_on_device(
                upscale_model,
                in_img,
                device,
                tag="SingleGPU",
                start_tile=tile_size,
                min_tile=min_tile_size,
                overlap=overlap,
            )
            return (out,)


    class ImageUpscaleWithModelMultiGPU:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "upscale_model": ("UPSCALE_MODEL",),
                    "image": ("IMAGE",),
                    "device_list": ("STRING", {"default": "auto"}),
                    "auto_max_devices": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                    "primary_share": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05}),
                    "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                    "min_tile_size": ("INT", {"default": 128, "min": 32, "max": 1024, "step": 32}),
                    "overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 4}),
                }
            }

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "upscale_multi"
        CATEGORY = "multiGPU_upscaler"

        def upscale_multi(
            self,
            upscale_model,
            image,
            device_list,
            auto_max_devices,
            primary_share,
            tile_size,
            min_tile_size,
            overlap,
        ):
            return (_multi_gpu_execute(
                upscale_model,
                image,
                device_list,
                auto_max_devices,
                primary_share,
                tile_size,
                min_tile_size,
                overlap,
            ),)


# ========== Shared multi-GPU logic (used by both APIs) ==========

def _multi_gpu_execute(
    upscale_model,
    image,
    device_list,
    auto_max_devices,
    primary_share,
    tile_size,
    min_tile_size,
    overlap,
):
    # Input -> [B, C, H, W]
    in_img = image.movedim(-1, -3)
    if in_img.ndim == 3:
        in_img = in_img.unsqueeze(0)
    b, c, h, w = in_img.shape

    if not torch.cuda.is_available():
        logging.info("[multiGPU] No CUDA, fallback to single GPU path.")
        device = model_management.get_torch_device()
        return run_tiled_on_device(
            upscale_model,
            in_img,
            device,
            tag="multiGPU-Single-NoCUDA",
            start_tile=tile_size,
            min_tile=min_tile_size,
            overlap=overlap,
        )

    if b < 2:
        logging.info("[multiGPU] Batch < 2, multi-GPU not beneficial. Single GPU.")
        device = model_management.get_torch_device()
        return run_tiled_on_device(
            upscale_model,
            in_img,
            device,
            tag="multiGPU-Single-SmallBatch",
            start_tile=tile_size,
            min_tile=min_tile_size,
            overlap=overlap,
        )

    all_devs = get_cuda_devices_sorted_free()
    if not all_devs:
        logging.info("[multiGPU] No CUDA devices detected; single GPU fallback.")
        device = model_management.get_torch_device()
        return run_tiled_on_device(
            upscale_model,
            in_img,
            device,
            tag="multiGPU-Single-NoDevs",
            start_tile=tile_size,
            min_tile=min_tile_size,
            overlap=overlap,
        )

    max_gpus = min(10, torch.cuda.device_count())
    explicit_indices = parse_device_list(device_list, max_gpus)

    if explicit_indices:
        use_indices = [i for i in explicit_indices if any(d[0] == i for d in all_devs)]
    else:
        auto_max = max(1, min(auto_max_devices, max_gpus))
        use_indices = [d[0] for d in all_devs[:auto_max]]

    use_indices = sorted(list(dict.fromkeys(use_indices)))
    if not use_indices:
        best = all_devs[0][0]
        dev = torch.device(f"cuda:{best}")
        logging.info(f"[multiGPU] No valid selection; using {dev}.")
        return run_tiled_on_device(
            upscale_model,
            in_img,
            dev,
            tag=f"multiGPU-Single-{dev}",
            start_tile=tile_size,
            min_tile=min_tile_size,
            overlap=overlap,
        )

    if len(use_indices) == 1:
        dev = torch.device(f"cuda:{use_indices[0]}")
        logging.info(f"[multiGPU] Only one GPU selected; using {dev}.")
        return run_tiled_on_device(
            upscale_model,
            in_img,
            dev,
            tag=f"multiGPU-Single-{dev}",
            start_tile=tile_size,
            min_tile=min_tile_size,
            overlap=overlap,
        )

    devices = [torch.device(f"cuda:{i}") for i in use_indices]
    logging.info(
        f"[multiGPU] Using GPUs {devices} for batch-parallel upscale. "
        f"Batch={b}, size={w}x{h}, tile={tile_size}, overlap={overlap}, "
        f"primary_share={primary_share}"
    )

    # Distribution: first GPU ~ primary_share, rest share remaining
    primary_count = max(1, min(b - 1, int(round(b * primary_share))))
    remaining = b - primary_count
    num_secondary = len(devices) - 1

    if remaining < num_secondary:
        devices = devices[: 1 + max(0, remaining)]
        num_secondary = len(devices) - 1

    if num_secondary <= 0:
        dev = devices[0]
        logging.info(f"[multiGPU] Degenerated to single GPU {dev}.")
        return run_tiled_on_device(
            upscale_model,
            in_img,
            dev,
            tag=f"multiGPU-Single-{dev}",
            start_tile=tile_size,
            min_tile=min_tile_size,
            overlap=overlap,
        )

    primary_count = max(1, min(b - num_secondary, int(round(b * primary_share))))
    remaining = b - primary_count
    base = remaining // num_secondary
    extra = remaining % num_secondary

    counts = [primary_count]
    for i in range(num_secondary):
        c_i = base + (1 if i < extra else 0)
        counts.append(c_i)

    total_assigned = sum(counts)
    if total_assigned != b:
        diff = b - total_assigned
        counts[0] += diff

    batches = []
    start = 0    # noqa: E305
    for c_i in counts:
        end = start + c_i
        batches.append(in_img[start:end])
        start = end

    results = [None] * len(devices)
    errors = [None] * len(devices)

    def worker(idx, model_desc, batch, device):
        tag = f"multiGPU-{device}"
        try:
            if batch.shape[0] == 0:
                results[idx] = torch.empty((0,), dtype=torch.float32)
                return
            results[idx] = run_tiled_on_device(
                model_desc,
                batch,
                device,
                tag=tag,
                start_tile=tile_size,
                min_tile=min_tile_size,
                overlap=overlap,
            )
        except Exception as e:
            logging.error(f"[{tag}] error: {e}")
            errors[idx] = e

    threads = []
    for i, device in enumerate(devices):
        t = threading.Thread(
            target=worker,
            args=(i, upscale_model, batches[i], device),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if any(errors):
        best = all_devs[0][0]
        dev = torch.device(f"cuda:{best}")
        logging.warning(
            "[multiGPU] One or more workers failed; falling back to "
            f"single GPU {dev}."
        )
        return run_tiled_on_device(
            upscale_model,
            in_img,
            dev,
            tag=f"multiGPU-FallbackSingle-{dev}",
            start_tile=tile_size,
            min_tile=min_tile_size,
            overlap=overlap,
        )

    final = torch.cat(results, dim=0)
    logging.info("[multiGPU] Completed multi-GPU batch-parallel upscale.")
    return final


# ========== Legacy loader exports ==========

# For classic ComfyUI custom node discovery:
NODE_CLASS_MAPPINGS = {
    "multiGPU_UpscaleModelLoader": UpscaleModelLoader,
    "multiGPU_ImageUpscaleWithModel": ImageUpscaleWithModel,
    "multiGPU_ImageUpscaleWithModelMultiGPU": ImageUpscaleWithModelMultiGPU,
}

NODES_LIST = list(NODE_CLASS_MAPPINGS.keys())