"""
Base Feature Module

This module provides the foundation for all features in the captioning system.
Each feature inherits from BaseFeature and provides:
- Default values and constraints
- Validation logic
- GUI component configuration
- Debug/logging methods
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """
    Configuration for a feature.
    
    Attributes:
        name: Unique identifier for the feature
        default_value: Default value when no override is provided
        min_value: Minimum allowed value (for numeric features)
        max_value: Maximum allowed value (for numeric features)
        description: Human-readable description
        gui_type: Type of Gradio component (slider, checkbox, textbox, dropdown, number)
        gui_label: Label shown in the UI
        gui_info: Tooltip/help text shown in the UI
        gui_step: Step size for sliders
        gui_choices: List of choices for dropdowns
    """
    name: str
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: str = ""
    gui_type: str = "slider"
    gui_label: str = ""
    gui_info: str = ""
    gui_step: Optional[float] = None
    gui_choices: Optional[List[str]] = None
    include_in_cli: bool = True  # Whether this feature should appear in CLI commands


class BaseFeature(ABC):
    """
    Abstract base class for all features.
    
    Each feature is self-contained with:
    - Default values
    - Validation logic  
    - GUI configuration
    - Debug output methods
    
    Example Usage:
        class TemperatureFeature(BaseFeature):
            @property
            def config(self) -> FeatureConfig:
                return FeatureConfig(
                    name="temperature",
                    default_value=0.7,
                    min_value=0.01,
                    max_value=2.0,
                    gui_type="slider",
                    gui_label="Temperature",
                    gui_info="Randomness: higher values are more creative."
                )
    """
    
    @property
    @abstractmethod
    def config(self) -> FeatureConfig:
        """Return feature configuration. Must be implemented by subclasses."""
        pass
    
    @property
    def name(self) -> str:
        """Return feature name."""
        return self.config.name
    
    def validate(self, value: Any) -> Any:
        """
        Validate and sanitize the input value.
        
        Override this method for custom validation logic.
        Default implementation handles min/max bounds.
        
        Args:
            value: The value to validate
            
        Returns:
            The validated (and possibly corrected) value
        """
        if value is None:
            self.log_debug(f"Value is None, using default: {self.config.default_value}")
            return self.config.default_value
            
        # Handle numeric bounds
        if self.config.min_value is not None:
            try:
                if value < self.config.min_value:
                    self.log_warning(f"Value {value} below minimum {self.config.min_value}, using minimum")
                    return self.config.min_value
            except TypeError:
                pass
                
        if self.config.max_value is not None:
            try:
                if value > self.config.max_value:
                    self.log_warning(f"Value {value} above maximum {self.config.max_value}, using maximum")
                    return self.config.max_value
            except TypeError:
                pass
                
        return value
    
    def get_default(self) -> Any:
        """Return the default value for this feature."""
        return self.config.default_value
    
    def get_validated_or_default(self, value: Any) -> Any:
        """
        Convenience method to get a validated value or the default.
        
        Args:
            value: Value to validate (can be None)
            
        Returns:
            Validated value or default if None
        """
        if value is None:
            return self.get_default()
        return self.validate(value)
    
    def log_debug(self, msg: str):
        """Log a debug message with feature context."""
        logger.debug(f"[{self.config.name}] {msg}")
    
    def log_info(self, msg: str):
        """Log an info message with feature context."""
        logger.info(f"[{self.config.name}] {msg}")
    
    def log_warning(self, msg: str):
        """Log a warning message with feature context."""
        logger.warning(f"[{self.config.name}] {msg}")
    
    def log_error(self, msg: str):
        """Log an error message with feature context."""
        logger.error(f"[{self.config.name}] {msg}")
    
    def get_gui_config(self) -> Dict[str, Any]:
        """
        Return configuration for building a Gradio component.
        
        Returns:
            Dictionary with keys: type, label, info, value, min, max, step, choices
        """
        return {
            "type": self.config.gui_type,
            "label": self.config.gui_label or self.config.name.replace("_", " ").title(),
            "info": self.config.gui_info or self.config.description,
            "value": self.config.default_value,
            "min": self.config.min_value,
            "max": self.config.max_value,
            "step": self.config.gui_step,
            "choices": self.config.gui_choices,
        }
    
    def apply_override(self, override_value: Any) -> 'BaseFeature':
        """
        Create a copy of this feature with an overridden default value.
        
        Note: This doesn't modify the original feature instance.
        
        Args:
            override_value: New default value
            
        Returns:
            Self (for method chaining) - override is stored separately
        """
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, default={self.config.default_value})"
