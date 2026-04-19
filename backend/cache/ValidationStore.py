"""
cache/ValidationStore.py
------------------------
Stores validated dataset input specs.

Validation IDs map to their original specs so /run/start can
reconstruct the data without re-parsing files or re-transmitting bytes.

Structure: _store[validation_id] -> (kind, spec_dict)
"""

import uuid
from typing import Dict, Tuple, Any

_store: Dict[str, Tuple[str, Dict[str, Any]]] = {}


def store_validation(kind: str, spec: Dict[str, Any]) -> str:
    """
    Store a validated spec. Returns a validation_id.
    
    Args:
        kind: 'synthetic' or 'upload'
        spec: validated specification dict
    
    Returns:
        validation_id (UUID string)
    """
    if not spec.get("name") or not spec.get("name").strip():
        raise ValueError("Spec must include a 'name' field")
    if _store and any(s[1].get("name") == spec["name"] for s in _store.values()):
        raise ValueError(f"Spec name '{spec['name']}' already exists")
    
    validation_id = str(uuid.uuid4())
    _store[validation_id] = (kind, spec)
    return validation_id


def get_validation(validation_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Retrieve a validated spec by its ID.
    
    Args:
        validation_id: UUID string
    
    Returns:
        (kind, spec) tuple
    
    Raises:
        KeyError if validation_id not found
    """
    if validation_id not in _store:
        raise KeyError(f"Validation ID '{validation_id}' not found")
    return _store[validation_id]


def delete_validation(validation_id: str) -> bool:
    """
    Delete a validated spec by its ID.
    
    Args:
        validation_id: UUID string
    
    Returns:
        True if deleted, False if not found
    """
    if validation_id in _store:
        del _store[validation_id]
        return True
    return False


def clear():
    """Clear all stored validations."""
    _store.clear()


def validation_ids() -> list:
    """List all stored validation IDs."""
    return list(_store.keys())