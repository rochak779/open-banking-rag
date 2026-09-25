"""Turn OBL OpenAPI documents into one chunk per API operation.

The spec is already structured, so chunking by operation preserves exactly the
unit a question is usually about ("what does POST /domestic-payments return?").
A generic character-window splitter would cut through the middle of an
operation and lose the endpoint name that makes the chunk findable at all.
"""

from pathlib import Path

import yaml

from obrag.models import Chunk

HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")

SPEC_REPO_URL = "https://github.com/OpenBankingUK/read-write-api-specs"


def _resolve(node: object, document: dict) -> object:
    """Follow a local '#/components/...' $ref; return anything else unchanged.

    The OBL specs put every parameter and response in components and point at
    them by $ref, so without this the operation text loses its names and
    descriptions entirely.
    """
    if isinstance(node, dict) and isinstance(node.get("$ref"), str):
        ref = node["$ref"]
        if ref.startswith("#/"):
            target: object = document
            for part in ref[2:].split("/"):
                target = target.get(part, {}) if isinstance(target, dict) else {}
            return target
    return node


def _schema_ref_name(node: object) -> str | None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            return ref.rsplit("/", 1)[-1]
    return None


def _describe_schema(name: str | None, schemas: dict) -> str:
    if not name:
        return ""
    description = schemas.get(name, {}).get("description", "")
    return f"{name}: {description}".strip().rstrip(":")


def _content_schemas(content: dict | None, schemas: dict) -> list[str]:
    """Describe each distinct schema in a content map, once.

    The OBL specs repeat the same schema under application/json,
    application/jose+jwe and a charset variant; listing all three is noise.
    """
    described: list[str] = []
    for media in (content or {}).values():
        text = _describe_schema(_schema_ref_name(media.get("schema")), schemas)
        if text and text not in described:
            described.append(text)
    return described


def _operation_text(
    title: str,
    version: str,
    method: str,
    path: str,
    operation: dict,
    document: dict,
) -> str:
    schemas = document.get("components", {}).get("schemas", {}) or {}
    lines = [
        f"{title} v{version}",
        f"Endpoint: {method} {path}",
    ]
    if operation.get("summary"):
        lines.append(f"Summary: {operation['summary']}")
    if operation.get("description"):
        lines.append(f"Description: {operation['description']}")

    parameters = [_resolve(p, document) for p in operation.get("parameters") or []]
    if parameters:
        lines.append("Parameters:")
        for parameter in parameters:
            required = "required" if parameter.get("required") else "optional"
            lines.append(
                f"  - {parameter.get('name')} (in {parameter.get('in')}, {required}): "
                f"{parameter.get('description', '')}".rstrip(": ")
            )

    body = _resolve(operation.get("requestBody", {}), document)
    for described in _content_schemas(body.get("content"), schemas):
        lines.append(f"Request body: {described}")

    responses = operation.get("responses") or {}
    if responses:
        lines.append("Responses:")
        for status, response in responses.items():
            response = _resolve(response, document)
            parts = [response.get("description", "")]
            parts += _content_schemas(response.get("content"), schemas)
            lines.append(f"  - {status}: " + " — ".join(p for p in parts if p))

    return "\n".join(lines)


def _spec_tag(path: Path, version: str) -> str:
    """The git tag the file was cloned from, read off fetch.py's directory name.

    The tag, not info.version, is what exists on GitHub: v4.0.1 is published
    under tags like v4.0.1-Update-1, so a link built from the version 404s.
    """
    for parent in path.parents:
        if parent.name.startswith("obl-specs-"):
            return parent.name.removeprefix("obl-specs-")
    return f"v{version}"


def parse_openapi_file(path: Path) -> list[Chunk]:
    document = yaml.safe_load(path.read_text())
    info = document.get("info", {})
    title = info.get("title", path.stem)
    version = str(info.get("version", "unknown"))
    source_url = f"{SPEC_REPO_URL}/blob/{_spec_tag(path, version)}/dist/openapi/{path.name}"
    slug = path.stem

    chunks: list[Chunk] = []
    for api_path, operations in (document.get("paths") or {}).items():
        for method_key, operation in operations.items():
            if method_key.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            method = method_key.upper()
            chunks.append(
                Chunk(
                    id=f"spec:{slug}:{method_key.lower()}:{api_path}",
                    text=_operation_text(title, version, method, api_path, operation, document),
                    collection="spec",
                    citation=f"{title} v{version} — {method} {api_path}",
                    source_url=source_url,
                    metadata={
                        "method": method,
                        "path": api_path,
                        "operation_id": operation.get("operationId", ""),
                        "api": title,
                        "api_version": version,
                    },
                )
            )
    return chunks


def parse_spec_dir(openapi_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(openapi_dir.glob("*.yaml")):
        chunks.extend(parse_openapi_file(path))
    return chunks
