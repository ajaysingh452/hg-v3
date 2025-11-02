"""Configuration management for Harmony Guard."""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from core.interfaces import ConfigurationManagerInterface


class ConfigurationManager(ConfigurationManagerInterface):
    """Manages configuration loading and updates for Harmony Guard."""
    
    def __init__(self, config_dir: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self._config_cache = {}
    
    def load_config(self, config_type: str, tenant_id: str = None) -> Dict[str, Any]:
        """
        Load configuration for specified type and tenant.
        
        Args:
            config_type: Type of configuration (ensemble, policy, preprocessing)
            tenant_id: Optional tenant identifier
            
        Returns:
            Configuration dictionary
        """
        cache_key = f"{config_type}_{tenant_id or 'default'}"
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Load base configuration
        base_config_file = self.config_dir / f"{config_type}.yaml"
        if config_type == "policy":
            base_config_file = self.config_dir / "policy_default.yaml"
            
        if not base_config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {base_config_file}")
        
        with open(base_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load tenant-specific overrides if available
        if tenant_id:
            tenant_config_file = self.config_dir / f"{config_type}_{tenant_id}.yaml"
            if tenant_config_file.exists():
                with open(tenant_config_file, 'r', encoding='utf-8') as f:
                    tenant_config = yaml.safe_load(f)
                config = self._merge_configs(config, tenant_config)
        
        # Cache the configuration
        self._config_cache[cache_key] = config
        return config
    
    def update_config(
        self, 
        config_type: str, 
        config_data: Dict[str, Any],
        tenant_id: str = None
    ) -> bool:
        """
        Update configuration for specified type and tenant.
        
        Args:
            config_type: Type of configuration
            config_data: New configuration data
            tenant_id: Optional tenant identifier
            
        Returns:
            Success status
        """
        try:
            if tenant_id:
                config_file = self.config_dir / f"{config_type}_{tenant_id}.yaml"
            else:
                config_file = self.config_dir / f"{config_type}.yaml"
                if config_type == "policy":
                    config_file = self.config_dir / "policy_default.yaml"
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)
            
            # Clear cache for this configuration
            cache_key = f"{config_type}_{tenant_id or 'default'}"
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
            
            return True
        except Exception as e:
            print(f"Error updating configuration: {e}")
            return False
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge tenant-specific configuration with base configuration.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_ensemble_config(self, tenant_id: str = None) -> Dict[str, Any]:
        """Get ensemble configuration."""
        return self.load_config("ensemble", tenant_id)
    
    def get_policy_config(self, tenant_id: str = None) -> Dict[str, Any]:
        """Get policy configuration."""
        return self.load_config("policy", tenant_id)
    
    def get_preprocessing_config(self, tenant_id: str = None) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.load_config("preprocessing", tenant_id)