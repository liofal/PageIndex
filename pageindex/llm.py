import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import openai

try:
    import boto3
except Exception:  # pragma: no cover - optional dependency
    boto3 = None


_DEFAULTS: Dict[str, Any] = {
    "provider": None,
    "aws_region": None,
    "aws_profile": None,
    "bedrock_inference_profile_arn": None,
    "bedrock_max_tokens": None,
    "openai_api_key": None,
}

_INFERENCE_PROFILE_ONLY = {
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
}


def set_defaults(
    provider: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_profile: Optional[str] = None,
    bedrock_inference_profile_arn: Optional[str] = None,
    bedrock_max_tokens: Optional[int] = None,
    openai_api_key: Optional[str] = None,
) -> None:
    def _normalize(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    provider = _normalize(provider)
    aws_region = _normalize(aws_region)
    aws_profile = _normalize(aws_profile)
    bedrock_inference_profile_arn = _normalize(bedrock_inference_profile_arn)

    if provider is not None:
        _DEFAULTS["provider"] = provider
    if aws_region is not None:
        _DEFAULTS["aws_region"] = aws_region
    if aws_profile is not None:
        _DEFAULTS["aws_profile"] = aws_profile
    if bedrock_inference_profile_arn is not None:
        _DEFAULTS["bedrock_inference_profile_arn"] = bedrock_inference_profile_arn
    if bedrock_max_tokens is not None:
        _DEFAULTS["bedrock_max_tokens"] = bedrock_max_tokens
    if openai_api_key is not None:
        _DEFAULTS["openai_api_key"] = openai_api_key


def get_defaults() -> Dict[str, Any]:
    defaults = dict(_DEFAULTS)
    if not defaults.get("provider"):
        defaults["provider"] = os.getenv("PAGEINDEX_PROVIDER") or None
    return defaults


def resolve_provider(provider: Optional[str], model: Optional[str]) -> str:
    if provider:
        return provider.lower()
    if model:
        lower = model.lower()
        if lower.startswith("eu."):
            return "bedrock"
        if any(lower.startswith(prefix) for prefix in (
            "anthropic.",
            "amazon.",
            "meta.",
            "mistral.",
            "cohere.",
            "google.",
            "openai.",
            "qwen.",
            "nvidia.",
        )):
            return "bedrock"
    return "openai"


def _default_aws_region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-west-1"


def _default_aws_profile() -> Optional[str]:
    return os.getenv("AWS_PROFILE") or None


def _default_bedrock_max_tokens() -> int:
    try:
        return int(os.getenv("BEDROCK_MAX_TOKENS", "2048"))
    except ValueError:
        return 2048


def _openai_client(api_key: Optional[str]) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key)


def _openai_async_client(api_key: Optional[str]) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=api_key)


def _bedrock_client(region: Optional[str], profile: Optional[str]):
    if boto3 is None:
        raise ImportError("boto3 is required for Bedrock support. Please install boto3.")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("bedrock-runtime", region_name=region or _default_aws_region())


def _messages_to_anthropic(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(str(block.get("text", "")))
            else:
                system_parts.append(str(content))
            continue

        if role not in ("user", "assistant"):
            role = "user"

        if isinstance(content, list):
            blocks: List[Dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    blocks.append({"type": "text", "text": str(block.get("text", ""))})
            if not blocks:
                blocks = [{"type": "text", "text": str(content)}]
        else:
            blocks = [{"type": "text", "text": str(content)}]

        converted.append({"role": role, "content": blocks})

    system = "\n\n".join([part for part in system_parts if part])
    return (system or None), converted


def _invoke_bedrock(
    messages: List[Dict[str, Any]],
    model_id: str,
    region: Optional[str],
    profile: Optional[str],
    inference_profile_arn: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
) -> Tuple[str, Optional[str]]:
    client = _bedrock_client(region, profile)
    if model_id.startswith(".eu."):
        model_id = "eu." + model_id[len(".eu."):]
    system, anthropic_messages = _messages_to_anthropic(messages)
    payload: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": anthropic_messages,
        "max_tokens": max_tokens or _default_bedrock_max_tokens(),
        "temperature": temperature,
    }
    if system:
        payload["system"] = system

    body = json.dumps(payload)
    model_id_to_use = inference_profile_arn or model_id
    model_ids_to_try = [model_id_to_use]
    model_id_stripped = model_id
    if model_id_stripped.startswith("eu."):
        model_id_stripped = model_id_stripped.split(".", 1)[1]
    if inference_profile_arn is None and model_id_stripped in _INFERENCE_PROFILE_ONLY:
        raise ValueError(
            "Bedrock model requires an inference profile. "
            "Set BEDROCK_INFERENCE_PROFILE_ARN or pass --bedrock-inference-profile-arn."
        )
    if inference_profile_arn is None and model_id and model_id.startswith("eu."):
        model_ids_to_try.append(model_id.split(".", 1)[1])

    last_error: Optional[Exception] = None
    for candidate in model_ids_to_try:
        try:
            response = client.invoke_model(
                modelId=candidate,
                body=body,
                accept="application/json",
                contentType="application/json",
            )
            raw_body = response.get("body")
            if hasattr(raw_body, "read"):
                raw_body = raw_body.read()
            data = json.loads(raw_body or "{}")
            text = ""
            if isinstance(data.get("content"), list):
                for block in data["content"]:
                    if block.get("type") == "text":
                        text += block.get("text", "")
            elif "completion" in data:
                text = data.get("completion", "")
            stop_reason = data.get("stop_reason") or data.get("stopReason")
            return text, stop_reason
        except Exception as exc:  # pragma: no cover - depends on AWS runtime
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise RuntimeError("Failed to invoke Bedrock model.")


def chat(
    messages: List[Dict[str, Any]],
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    region: Optional[str] = None,
    profile: Optional[str] = None,
    inference_profile_arn: Optional[str] = None,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    resolved_provider = resolve_provider(provider, model)
    if resolved_provider == "openai":
        client = _openai_client(api_key)
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content, response.choices[0].finish_reason
    if resolved_provider == "bedrock":
        if inference_profile_arn is None:
            inference_profile_arn = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN") or None
        return _invoke_bedrock(
            messages=messages,
            model_id=model,
            region=region,
            profile=profile,
            inference_profile_arn=inference_profile_arn,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported provider: {resolved_provider}")


async def chat_async(
    messages: List[Dict[str, Any]],
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    region: Optional[str] = None,
    profile: Optional[str] = None,
    inference_profile_arn: Optional[str] = None,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    resolved_provider = resolve_provider(provider, model)
    if resolved_provider == "openai":
        async with _openai_async_client(api_key) as client:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs["temperature"] = temperature
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content, response.choices[0].finish_reason
    if resolved_provider == "bedrock":
        if inference_profile_arn is None:
            inference_profile_arn = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN") or None
        return await asyncio.to_thread(
            _invoke_bedrock,
            messages,
            model,
            region,
            profile,
            inference_profile_arn,
            temperature,
            max_tokens,
        )
    raise ValueError(f"Unsupported provider: {resolved_provider}")
