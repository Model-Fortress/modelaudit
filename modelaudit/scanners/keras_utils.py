"""Shared utilities for Keras scanners."""

import base64
from typing import Any

from modelaudit.detectors.suspicious_symbols import KNOWN_SAFE_MODEL_CLASSES

from .base import IssueSeverity, ScanResult

# Dangerous patterns to look for inside decoded Lambda bytecode / source
_LAMBDA_DANGEROUS_PATTERNS: list[str] = [
    "exec",
    "eval",
    "__import__",
    "compile",
    "open",
    "subprocess",
    "os.system",
    "os.popen",
    "pickle",
    "marshal",
    "importlib",
    "runpy",
    "webbrowser",
    "socket",
    "http",
    "urllib",
    "shutil",
    "ctypes",
]


def check_lambda_dict_function(
    function_dict: dict[str, Any],
    result: ScanResult,
    location: str,
    layer_name: str,
) -> bool:
    """Check a Keras 3.x dict-format Lambda function for dangerous code.

    Keras 3.x serialises Lambda functions as::

        {"class_name": "__lambda__",
         "config": {"code": "<base64-encoded bytecode>", ...}}

    Returns True if the dict was recognised and handled, False otherwise.
    """
    class_name = function_dict.get("class_name")
    if class_name != "__lambda__":
        return False

    config = function_dict.get("config")
    if not isinstance(config, dict):
        result.add_check(
            name="Lambda Layer Detection",
            passed=False,
            message=f"Lambda layer '{layer_name}' uses malformed dict-format function metadata (config is not a dict)",
            severity=IssueSeverity.WARNING,
            location=location,
            details={
                "layer_name": layer_name,
                "layer_class": "Lambda",
                "function_format": "dict",
                "parse_status": "invalid_config",
                "function_dict": function_dict,
            },
            why="Malformed dict-format Lambda metadata is suspicious and prevents bytecode inspection.",
        )
        return True

    code_b64 = config.get("code")
    if not code_b64 or not isinstance(code_b64, str):
        # Dict-format Lambda with no code field — flag as suspicious
        result.add_check(
            name="Lambda Layer Detection",
            passed=False,
            message=f"Lambda layer '{layer_name}' uses dict-format function with no code field",
            severity=IssueSeverity.WARNING,
            location=location,
            details={
                "layer_name": layer_name,
                "layer_class": "Lambda",
                "function_format": "dict",
            },
            why="Lambda layers with dict-format functions indicate Keras 3.x bytecode serialisation.",
        )
        return True

    try:
        decoded = base64.b64decode(code_b64)
        decoded_str = decoded.decode("utf-8", errors="replace")
    except Exception:
        result.add_check(
            name="Lambda Layer Detection",
            passed=False,
            message=f"Lambda layer '{layer_name}' contains non-decodable dict-format code",
            severity=IssueSeverity.WARNING,
            location=location,
            details={
                "layer_name": layer_name,
                "layer_class": "Lambda",
                "function_format": "dict",
            },
            why="Unable to decode Lambda bytecode for security analysis.",
        )
        return True

    found_patterns = [p for p in _LAMBDA_DANGEROUS_PATTERNS if p in decoded_str.lower()]

    if found_patterns:
        result.add_check(
            name="Lambda Layer Code Analysis",
            passed=False,
            message=(
                f"Lambda layer '{layer_name}' contains dangerous patterns in bytecode: {', '.join(found_patterns)}"
            ),
            severity=IssueSeverity.CRITICAL,
            location=location,
            details={
                "layer_name": layer_name,
                "layer_class": "Lambda",
                "dangerous_patterns": found_patterns,
                "function_format": "dict_bytecode",
                "code_preview": decoded_str[:200] + "..." if len(decoded_str) > 200 else decoded_str,
            },
            why=(
                "Lambda layers can execute arbitrary Python code during model inference. "
                "Dangerous patterns found in the embedded bytecode."
            ),
        )
    else:
        result.add_check(
            name="Lambda Layer Code Analysis",
            passed=False,
            message=f"Lambda layer '{layer_name}' contains embedded bytecode (dict-format)",
            severity=IssueSeverity.WARNING,
            location=location,
            details={
                "layer_name": layer_name,
                "layer_class": "Lambda",
                "function_format": "dict_bytecode",
            },
            why=(
                "Keras 3.x Lambda layers embed compiled bytecode that will execute "
                "arbitrary code during model loading or inference."
            ),
        )
    return True


def check_subclassed_model(
    model_class: str,
    result: ScanResult,
    location: str,
) -> None:
    """Check whether a Keras model class is subclassed (custom) or a known safe class.

    Subclassed models can contain arbitrary Python code in their call() method,
    unlike standard Keras models that use declarative layer configurations.

    Args:
        model_class: The class_name from the Keras model config.
        result: ScanResult to add the check to.
        location: File path for the check location.
    """
    if model_class and model_class not in KNOWN_SAFE_MODEL_CLASSES:
        result.add_check(
            name="Subclassed Model Detection",
            passed=False,
            message=f"Subclassed Keras model detected: {model_class}",
            severity=IssueSeverity.INFO,
            location=location,
            details={
                "model_class": model_class,
                "known_safe_classes": sorted(KNOWN_SAFE_MODEL_CLASSES),
                "risk": "Subclassed models require external Python code to load, which should be reviewed",
            },
            why=(
                "Subclassed Keras models (custom class names) require external Python class "
                "definitions to load. The model file itself does not contain executable code, "
                "but the loading code should be reviewed. Standard Keras models (Sequential, "
                "Functional, Model) use declarative layer configurations and load without custom code."
            ),
        )
    elif model_class in KNOWN_SAFE_MODEL_CLASSES:
        result.add_check(
            name="Subclassed Model Detection",
            passed=True,
            message=f"Standard Keras model class: {model_class}",
            location=location,
            details={"model_class": model_class},
        )
