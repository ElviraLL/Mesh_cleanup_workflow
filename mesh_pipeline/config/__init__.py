"""Config loading and validation for the mesh-cleanup pipeline.

See mesh_pipeline/config/schema.py for the default config, load_config(),
and validation rules; mesh_pipeline/config/humanoid.yaml for the shipped
defaults file (PLAN.md's config block, in valid YAML).
"""

from mesh_pipeline.config.schema import DEFAULT_CONFIG, load_config

__all__ = ["DEFAULT_CONFIG", "load_config"]
