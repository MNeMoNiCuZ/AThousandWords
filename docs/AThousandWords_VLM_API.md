# API Usage Guide

## Starting the server

```
server.bat
```

This starts the API on `http://localhost:8585` (equivalent to `gui.bat --server --port 8585`).

## Endpoints

Each endpoint is available both with and without an `/api` prefix (e.g. `/api/health` and `/health` both work).

### `GET /health`

Health check.

```
curl http://localhost:8585/health
```

Response:

```json
{"status": "ok"}
```

### `GET /models`

Lists available model IDs and per-model batch-size info.

```
curl http://localhost:8585/models
```

Response:

```json
{
  "models": ["model_id_1", "model_id_2", ...],
  "gpu_vram": 24,
  "batch_sizes": {
    "model_id_1": {
      "default": 1,
      "recommended": 4,
      "choices": [1, 2, 4, 8],
      "min": 1,
      "max": null
    }
  }
}
```

### `POST /caption`

Submit one or more images for captioning. `multipart/form-data` request.

Form fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `files` | file(s) | yes | One or more image files |
| `model` | string | yes | Must be one of the IDs from `GET /models` |
| `batch_size` | int | no | Defaults to the VRAM-recommended value if omitted |
| `task_prompt` | string | no | |
| `max_tokens` | int | no | |
| `temperature` | float | no | |

Example:

```
curl -X POST http://localhost:8585/caption \
  -F "model=model_id_1" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

Response:

```json
{
  "status": "success",
  "results": [
    {"filename": "image1.jpg", "caption": "..."},
    {"filename": "image2.jpg", "caption": "..."}
  ]
}
```

Errors return an HTTP error status with a JSON `detail` field (e.g. `400` for an unknown model or no valid images, `500` for inference failures).

## Captioning video

`POST /caption` also accepts video files in `files` alongside or instead of images. Supported extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`. Use a model that supports video input.

```
curl -X POST http://localhost:8585/caption \
  -F "model=model_id_1" \
  -F "files=@clip.mp4"
```

The API does not expose a `fps` form field, so video frame sampling rate is not overridable per-request — it uses the model's/global configured default (see `fps` feature, default 4 FPS, range 1-60). To change it, adjust the model's or global config before starting the server.
