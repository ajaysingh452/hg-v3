"""Policy configuration loader with tenant-specific overrides."""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolicyProfile:
    """Policy profile configuration."""
    name: str
    version: str
    block_thresholds: Dict[str, Dict[str, float]]
    safe_contexts: List[str]
    department_overrides: Dict[str, Dict[str, Any]]
    hard_rules: List[Dict[str, Any]]
    tenant_lexicon_overlay: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ThresholdConfig:
    """Threshold configuration for a category."""
    category: str
    medium: float
    high: float
    critical: float


@dataclass
class DepartmentOverride:
    """Department-specific policy overrides."""
    department: str
    allow_sensitive_discussions: bool = False
    allow_policy_examples: bool = False
    allow_threat_analysis: bool = False
    review_threshold_adjustment: float = 0.0


@dataclass
class HardRule:
    """Hard policy rule that cannot be overridden."""
    category: str
    severity: str
    action: str
    override_allowed: bool = False


class PolicyConfigLoader:
    """Loads and manages policy configurations with tenant-specific overrides."""
    
    def __init__(self, config_dir: str = None):
        """
        Initialize policy configuration loader.
        
        Args:
            config_dir: Directory containing policy configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        self.config_dir = Path(config_dir)
        self._config_cache = {}
        self._tenant_configs = {}
        
    def load_policy_profile(self, tenant_id: str = None) -> PolicyProfile:
        """
        Load policy profile for specified tenant.
        
        Args:
            tenant_id: Optional tenant identifier
            
        Returns:
            PolicyProfile configuration
        """
        cache_key = f"policy_{tenant_id or 'default'}"
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Load base policy configuration
        base_config = self._load_base_policy()
        
        # Apply tenant-specific overrides if available
        if tenant_id:
            tenant_overrides = self._load_tenant_overrides(tenant_id)
            if tenant_overrides:
                base_config = self._merge_policy_configs(base_config, tenant_overrides)
        
        # Convert to PolicyProfile object
        profile = self._dict_to_policy_profile(base_config)
        
        # Cache the configuration
        self._config_cache[cache_key] = profile
        return profile
    
    def _load_base_policy(self) -> Dict[str, Any]:
        """Load base policy configuration."""
        base_config_file = self.config_dir / "policy_default.yaml"
        
        if not base_config_file.exists():
            raise FileNotFoundError(f"Base policy configuration not found: {base_config_file}")
        
        with open(base_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('policy_profile', {})
    
    def _load_tenant_overrides(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Load tenant-specific policy overrides.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Tenant override configuration or None if not found
        """
        tenant_config_file = self.config_dir / f"policy_{tenant_id}.yaml"
        
        if not tenant_config_file.exists():
            logger.debug(f"No tenant-specific policy found for {tenant_id}")
            return None
        
        try:
            with open(tenant_config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('policy_profile', {})
        except Exception as e:
            logger.error(f"Error loading tenant policy for {tenant_id}: {e}")
            return None
    
    def _merge_policy_configs(
        self, 
        base_config: Dict[str, Any], 
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge tenant-specific overrides with base policy configuration.
        
        Args:
            base_config: Base policy configuration
            override_config: Tenant-specific overrides
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        # Merge block thresholds
        if 'block_thresholds' in override_config:
            if 'block_thresholds' not in merged:
                merged['block_thresholds'] = {}
            for category, thresholds in override_config['block_thresholds'].items():
                if category in merged['block_thresholds']:
                    merged['block_thresholds'][category].update(thresholds)
                else:
                    merged['block_thresholds'][category] = thresholds
        
        # Merge safe contexts (union of both lists)
        if 'safe_contexts' in override_config:
            base_contexts = set(merged.get('safe_contexts', []))
            override_contexts = set(override_config['safe_contexts'])
            merged['safe_contexts'] = list(base_contexts.union(override_contexts))
        
        # Merge department overrides
        if 'department_overrides' in override_config:
            if 'department_overrides' not in merged:
                merged['department_overrides'] = {}
            for dept, overrides in override_config['department_overrides'].items():
                if dept in merged['department_overrides']:
                    merged['department_overrides'][dept].update(overrides)
                else:
                    merged['department_overrides'][dept] = overrides
        
        # Hard rules cannot be overridden, but additional ones can be added
        if 'hard_rules' in override_config:
            base_rules = merged.get('hard_rules', [])
            additional_rules = override_config['hard_rules']
            merged['hard_rules'] = base_rules + additional_rules
        
        # Other fields can be directly overridden
        for key in ['name', 'version', 'tenant_lexicon_overlay']:
            if key in override_config:
                merged[key] = override_config[key]
        
        return merged
    
    def _dict_to_policy_profile(self, config_dict: Dict[str, Any]) -> PolicyProfile:
        """
        Convert configuration dictionary to PolicyProfile object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            PolicyProfile object
        """
        return PolicyProfile(
            name=config_dict.get('name', 'default'),
            version=config_dict.get('version', '1.0'),
            block_thresholds=config_dict.get('block_thresholds', {}),
            safe_contexts=config_dict.get('safe_contexts', []),
            department_overrides=config_dict.get('department_overrides', {}),
            hard_rules=config_dict.get('hard_rules', []),
            tenant_lexicon_overlay=config_dict.get('tenant_lexicon_overlay'),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def save_tenant_policy(self, tenant_id: str, policy_profile: PolicyProfile) -> bool:
        """
        Save tenant-specific policy configuration.
        
        Args:
            tenant_id: Tenant identifier
            policy_profile: Policy profile to save
            
        Returns:
            Success status
        """
        try:
            tenant_config_file = self.config_dir / f"policy_{tenant_id}.yaml"
            
            # Convert PolicyProfile to dictionary
            config_dict = {
                'policy_profile': {
                    'name': policy_profile.name,
                    'version': policy_profile.version,
                    'block_thresholds': policy_profile.block_thresholds,
                    'safe_contexts': policy_profile.safe_contexts,
                    'department_overrides': policy_profile.department_overrides,
                    'hard_rules': policy_profile.hard_rules,
                    'tenant_lexicon_overlay': policy_profile.tenant_lexicon_overlay,
                    'updated_at': datetime.now().isoformat()
                }
            }
            
            with open(tenant_config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
            
            # Clear cache for this tenant
            cache_key = f"policy_{tenant_id}"
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
            
            logger.info(f"Saved tenant policy configuration for {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving tenant policy for {tenant_id}: {e}")
            return False
    
    def get_threshold_config(self, category: str, tenant_id: str = None) -> Optional[ThresholdConfig]:
        """
        Get threshold configuration for specific category.
        
        Args:
            category: Abuse category
            tenant_id: Optional tenant identifier
            
        Returns:
            ThresholdConfig or None if not found
        """
        profile = self.load_policy_profile(tenant_id)
        
        if category not in profile.block_thresholds:
            return None
        
        thresholds = profile.block_thresholds[category]
        return ThresholdConfig(
            category=category,
            medium=thresholds.get('medium', 0.8),
            high=thresholds.get('high', 0.6),
            critical=thresholds.get('critical', 0.4)
        )
    
    def get_department_override(
        self, 
        department: str, 
        tenant_id: str = None
    ) -> Optional[DepartmentOverride]:
        """
        Get department-specific overrides.
        
        Args:
            department: Department name
            tenant_id: Optional tenant identifier
            
        Returns:
            DepartmentOverride or None if not found
        """
        profile = self.load_policy_profile(tenant_id)
        
        if department not in profile.department_overrides:
            return None
        
        overrides = profile.department_overrides[department]
        return DepartmentOverride(
            department=department,
            allow_sensitive_discussions=overrides.get('allow_sensitive_discussions', False),
            allow_policy_examples=overrides.get('allow_policy_examples', False),
            allow_threat_analysis=overrides.get('allow_threat_analysis', False),
            review_threshold_adjustment=overrides.get('review_threshold_adjustment', 0.0)
        )
    
    def get_hard_rules(self, tenant_id: str = None) -> List[HardRule]:
        """
        Get hard rules that cannot be overridden.
        
        Args:
            tenant_id: Optional tenant identifier
            
        Returns:
            List of HardRule objects
        """
        profile = self.load_policy_profile(tenant_id)
        
        hard_rules = []
        for rule_dict in profile.hard_rules:
            hard_rules.append(HardRule(
                category=rule_dict['category'],
                severity=rule_dict['severity'],
                action=rule_dict['action'],
                override_allowed=rule_dict.get('override_allowed', False)
            ))
        
        return hard_rules
    
    def clear_cache(self, tenant_id: str = None):
        """
        Clear configuration cache.
        
        Args:
            tenant_id: Optional tenant identifier to clear specific cache
        """
        if tenant_id:
            cache_key = f"policy_{tenant_id}"
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
        else:
            self._config_cache.clear()