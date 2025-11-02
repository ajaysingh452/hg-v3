"""Intent and context analysis module for Harmony Guard."""

from .context_analyzer import ContextAnalyzer

# Create a main IntentContextLayer class that wraps the ContextAnalyzer
class IntentContextLayer:
    """Main Intent/Context Layer for the Harmony Guard ensemble."""
    
    def __init__(self, config: dict = None):
        """Initialize the Intent/Context Layer."""
        self.config = config or {}
        self.context_analyzer = ContextAnalyzer()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Intent/Context Layer."""
        # Any async initialization can go here
        self._initialized = True
    
    async def analyze_context(self, processed_text, lpe_result, classifier_result):
        """Analyze context and return ContextResult."""
        if not self._initialized:
            raise RuntimeError("IntentContextLayer not initialized")
        
        return self.context_analyzer.analyze_context(
            processed_text, lpe_result, classifier_result
        )
    
    async def is_healthy(self):
        """Check if the component is healthy."""
        return self._initialized
    
    async def shutdown(self):
        """Shutdown the component."""
        self._initialized = False

__all__ = ['IntentContextLayer', 'ContextAnalyzer']