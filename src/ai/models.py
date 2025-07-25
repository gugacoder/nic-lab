"""
Groq Model Configuration and Constants

Defines available models, their capabilities, pricing, and usage patterns
for the NIC Chat system's AI integration.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Model size categories"""
    SMALL = "small"      # < 10B parameters
    MEDIUM = "medium"    # 10B - 30B parameters  
    LARGE = "large"      # 30B - 100B parameters
    EXTRA_LARGE = "xl"   # > 100B parameters


class ModelCapability(Enum):
    """Model capability flags"""
    CHAT = "chat"                    # Conversational AI
    CODE = "code"                    # Code generation and analysis
    REASONING = "reasoning"          # Complex reasoning tasks
    SUMMARIZATION = "summarization"  # Text summarization
    TRANSLATION = "translation"     # Language translation
    CREATIVE = "creative"            # Creative writing
    ANALYSIS = "analysis"           # Document analysis
    MATH = "math"                   # Mathematical reasoning


@dataclass
class ModelPricing:
    """Model pricing information (per million tokens)"""
    input_cost: float    # Cost per million input tokens
    output_cost: float   # Cost per million output tokens
    currency: str = "USD"


@dataclass
class ModelLimits:
    """Model operational limits"""
    max_tokens: int              # Maximum tokens per request
    context_window: int          # Total context window size
    max_requests_per_minute: int # API rate limit
    max_tokens_per_minute: int   # Token-based rate limit


@dataclass 
class ModelInfo:
    """Complete model information"""
    id: str
    name: str
    provider: str
    size: ModelSize
    capabilities: List[ModelCapability]
    limits: ModelLimits
    pricing: ModelPricing
    description: str
    recommended_use_cases: List[str]
    strengths: List[str]
    limitations: List[str]
    release_date: Optional[str] = None
    is_default: bool = False
    is_deprecated: bool = False


# Groq Model Definitions
GROQ_MODELS = {
    "llama-3.1-405b-reasoning": ModelInfo(
        id="llama-3.1-405b-reasoning",
        name="Llama 3.1 405B Reasoning",
        provider="groq",
        size=ModelSize.EXTRA_LARGE,
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.REASONING,
            ModelCapability.CODE,
            ModelCapability.ANALYSIS,
            ModelCapability.MATH,
            ModelCapability.SUMMARIZATION
        ],
        limits=ModelLimits(
            max_tokens=8192,
            context_window=131072,  # 128k context
            max_requests_per_minute=30,
            max_tokens_per_minute=6000
        ),
        pricing=ModelPricing(
            input_cost=5.32,   # $5.32 per 1M tokens
            output_cost=16.00  # $16.00 per 1M tokens
        ),
        description="Most capable Llama model with advanced reasoning capabilities",
        recommended_use_cases=[
            "Complex reasoning tasks",
            "Advanced code generation", 
            "Mathematical problem solving",
            "Research and analysis",
            "Multi-step planning"
        ],
        strengths=[
            "Exceptional reasoning abilities",
            "Strong code generation",
            "Large context window",
            "Excellent instruction following"
        ],
        limitations=[
            "Higher cost per token",
            "Slower inference speed",
            "Limited availability"
        ],
        release_date="2024-07"
    ),
    
    "llama-3.1-70b-versatile": ModelInfo(
        id="llama-3.1-70b-versatile", 
        name="Llama 3.1 70B Versatile",
        provider="groq",
        size=ModelSize.LARGE,
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.SUMMARIZATION,
            ModelCapability.CREATIVE,
            ModelCapability.ANALYSIS
        ],
        limits=ModelLimits(
            max_tokens=8192,
            context_window=131072,  # 128k context
            max_requests_per_minute=30,
            max_tokens_per_minute=6000
        ),
        pricing=ModelPricing(
            input_cost=0.59,   # $0.59 per 1M tokens
            output_cost=0.79   # $0.79 per 1M tokens
        ),
        description="Balanced model optimized for versatile use cases",
        recommended_use_cases=[
            "General conversation",
            "Document analysis",
            "Code assistance",
            "Content generation",
            "Question answering"
        ],
        strengths=[
            "Excellent price/performance ratio",
            "Fast inference speed", 
            "Large context window",
            "Versatile capabilities"
        ],
        limitations=[
            "Less reasoning capability than 405B",
            "May struggle with very complex tasks"
        ],
        is_default=False,
        release_date="2024-07"
    ),
    
    "llama-3.1-8b-instant": ModelInfo(
        id="llama-3.1-8b-instant",
        name="Llama 3.1 8B Instant", 
        provider="groq",
        size=ModelSize.SMALL,
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.SUMMARIZATION,
            ModelCapability.CREATIVE,
            ModelCapability.CODE
        ],
        limits=ModelLimits(
            max_tokens=8192,
            context_window=131072,  # 128k context
            max_requests_per_minute=30,
            max_tokens_per_minute=30000
        ),
        pricing=ModelPricing(
            input_cost=0.05,   # $0.05 per 1M tokens
            output_cost=0.08   # $0.08 per 1M tokens
        ),
        description="Fast, cost-effective model for simple tasks",
        recommended_use_cases=[
            "Simple conversations",
            "Quick summaries",
            "Basic code completion",
            "Fast prototyping",
            "High-volume applications"
        ],
        strengths=[
            "Extremely fast inference",
            "Very low cost",
            "High throughput",
            "Good for simple tasks"
        ],
        limitations=[
            "Limited reasoning capabilities",
            "Less nuanced responses",
            "May struggle with complex queries"
        ],
        is_default=True,
        release_date="2024-07"
    ),
    
    "mixtral-8x7b-32768": ModelInfo(
        id="mixtral-8x7b-32768",
        name="Mixtral 8x7B",
        provider="groq", 
        size=ModelSize.MEDIUM,
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.TRANSLATION,
            ModelCapability.SUMMARIZATION
        ],
        limits=ModelLimits(
            max_tokens=32768,
            context_window=32768,
            max_requests_per_minute=30,
            max_tokens_per_minute=5000
        ),
        pricing=ModelPricing(
            input_cost=0.24,   # $0.24 per 1M tokens
            output_cost=0.24   # $0.24 per 1M tokens
        ),
        description="Efficient mixture-of-experts model with strong performance",
        recommended_use_cases=[
            "Multilingual tasks",
            "Code generation",
            "Technical documentation",
            "Translation tasks",
            "Balanced performance needs"
        ],
        strengths=[
            "Strong multilingual capabilities",
            "Good code performance",
            "Efficient architecture",
            "Balanced cost/capability"
        ],
        limitations=[
            "Smaller context window than Llama 3.1",
            "May have inconsistencies across languages"
        ],
        release_date="2023-12"
    )
}


class ModelSelector:
    """Intelligent model selection based on task requirements"""
    
    def __init__(self):
        self.models = GROQ_MODELS
        logger.info(f"Initialized ModelSelector with {len(self.models)} models")
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID"""
        return self.models.get(model_id)
    
    def get_default_model(self) -> ModelInfo:
        """Get the default model"""
        for model in self.models.values():
            if model.is_default:
                return model
        # Fallback to first available model
        return next(iter(self.models.values()))
    
    def list_models(
        self,
        capability: Optional[ModelCapability] = None,
        size: Optional[ModelSize] = None,
        max_cost: Optional[float] = None
    ) -> List[ModelInfo]:
        """List models with optional filtering"""
        models = list(self.models.values())
        
        # Filter by capability
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        # Filter by size
        if size:
            models = [m for m in models if m.size == size]
        
        # Filter by cost
        if max_cost:
            models = [m for m in models if m.pricing.output_cost <= max_cost]
        
        # Sort by performance (larger models first, then by cost)
        size_order = {ModelSize.EXTRA_LARGE: 4, ModelSize.LARGE: 3, 
                     ModelSize.MEDIUM: 2, ModelSize.SMALL: 1}
        
        return sorted(models, key=lambda m: (size_order[m.size], -m.pricing.output_cost), reverse=True)
    
    def recommend_model(
        self,
        task_type: str,
        cost_sensitive: bool = False,
        speed_priority: bool = False,
        context_length: Optional[int] = None
    ) -> ModelInfo:
        """Recommend best model for a given task"""
        
        # Map task types to capabilities
        task_capability_map = {
            "conversation": ModelCapability.CHAT,
            "code": ModelCapability.CODE,
            "analysis": ModelCapability.ANALYSIS,
            "reasoning": ModelCapability.REASONING,
            "summary": ModelCapability.SUMMARIZATION,
            "creative": ModelCapability.CREATIVE,
            "translation": ModelCapability.TRANSLATION,
            "math": ModelCapability.MATH
        }
        
        capability = task_capability_map.get(task_type, ModelCapability.CHAT)
        models = self.list_models(capability=capability)
        
        if not models:
            return self.get_default_model()
        
        # Filter by context length requirement
        if context_length:
            models = [m for m in models if m.limits.context_window >= context_length]
            if not models:
                logger.warning(f"No models support required context length {context_length}")
                return self.get_default_model()
        
        # Apply selection criteria
        if speed_priority:
            # Prefer smaller, faster models
            models.sort(key=lambda m: (m.size.value, m.pricing.output_cost))
            return models[0]
        elif cost_sensitive:
            # Prefer cheapest models that meet requirements
            models.sort(key=lambda m: m.pricing.output_cost)
            return models[0]
        else:
            # Balance performance and cost (default to 70B unless specific need)
            preferred_models = [m for m in models if m.id == "llama-3.1-8b-instant"]
            if preferred_models:
                return preferred_models[0]
            return models[0] if models else self.get_default_model()
    
    def estimate_cost(
        self, 
        model_id: str, 
        input_tokens: int, 
        output_tokens: int
    ) -> Dict[str, float]:
        """Estimate cost for a request"""
        model = self.get_model(model_id)
        if not model:
            return {"error": "Model not found"}
        
        input_cost = (input_tokens / 1_000_000) * model.pricing.input_cost
        output_cost = (output_tokens / 1_000_000) * model.pricing.output_cost
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost, 
            "total_cost": total_cost,
            "currency": model.pricing.currency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all available models"""
        summary = {
            "total_models": len(self.models),
            "by_size": {},
            "by_capability": {},
            "default_model": None,
            "cost_range": {"min": float('inf'), "max": 0}
        }
        
        # Aggregate by size
        for model in self.models.values():
            size_key = model.size.value
            summary["by_size"][size_key] = summary["by_size"].get(size_key, 0) + 1
            
            # Track cost range
            summary["cost_range"]["min"] = min(summary["cost_range"]["min"], model.pricing.output_cost)
            summary["cost_range"]["max"] = max(summary["cost_range"]["max"], model.pricing.output_cost)
            
            # Find default
            if model.is_default:
                summary["default_model"] = model.id
        
        # Aggregate by capability
        for model in self.models.values():
            for capability in model.capabilities:
                cap_key = capability.value
                summary["by_capability"][cap_key] = summary["by_capability"].get(cap_key, 0) + 1
        
        return summary


# Global model selector instance
model_selector = ModelSelector()

# Convenience functions
def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get model information"""
    return model_selector.get_model(model_id)

def get_default_model() -> ModelInfo:
    """Get default model"""
    return model_selector.get_default_model()

def recommend_model(**kwargs) -> ModelInfo:
    """Recommend best model for task"""
    return model_selector.recommend_model(**kwargs)

def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
    """Estimate request cost"""
    return model_selector.estimate_cost(model_id, input_tokens, output_tokens)


# Model validation
def validate_model_config():
    """Validate model configuration consistency"""
    issues = []
    
    for model_id, model in GROQ_MODELS.items():
        # Check ID consistency
        if model_id != model.id:
            issues.append(f"Model ID mismatch: {model_id} != {model.id}")
        
        # Check required fields
        if not model.name or not model.description:
            issues.append(f"Model {model_id} missing required fields")
        
        # Check pricing
        if model.pricing.input_cost <= 0 or model.pricing.output_cost <= 0:
            issues.append(f"Model {model_id} has invalid pricing")
        
        # Check limits
        if model.limits.max_tokens <= 0 or model.limits.context_window <= 0:
            issues.append(f"Model {model_id} has invalid limits")
    
    # Check for default model
    default_models = [m for m in GROQ_MODELS.values() if m.is_default]
    if len(default_models) != 1:
        issues.append(f"Expected exactly 1 default model, found {len(default_models)}")
    
    return issues


if __name__ == "__main__":
    # Model configuration validation and testing
    print("Groq Model Configuration")
    print("=" * 40)
    
    # Validate configuration
    issues = validate_model_config()
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Configuration valid")
    
    print(f"\nAvailable Models: {len(GROQ_MODELS)}")
    
    # Show model summary
    summary = model_selector.get_model_summary()
    print(f"Default Model: {summary['default_model']}")
    print(f"Cost Range: ${summary['cost_range']['min']:.3f} - ${summary['cost_range']['max']:.3f} per 1M tokens")
    
    print("\nModels by Size:")
    for size, count in summary["by_size"].items():
        print(f"  {size}: {count}")
    
    print("\nModels by Capability:")
    for cap, count in summary["by_capability"].items():
        print(f"  {cap}: {count}")
    
    # Test recommendations
    print("\nRecommendations:")
    print(f"  General use: {recommend_model(task_type='conversation').name}")
    print(f"  Code tasks: {recommend_model(task_type='code').name}")
    print(f"  Cost-sensitive: {recommend_model(task_type='conversation', cost_sensitive=True).name}")
    print(f"  Speed priority: {recommend_model(task_type='conversation', speed_priority=True).name}")