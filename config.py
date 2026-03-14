from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class ThinkingConfig(BaseModel):
    """Thinking configuration for models"""

    type: str = Field(..., description="Type of thinking: enabled/disabled")
    budget_tokens: int = Field(
        alias="budgetTokens", default=1024, description="Budget tokens for thinking"
    )


class Modalities(BaseModel):
    """Input and output modalities for a model"""

    input: List[str] = Field(
        default_factory=list, description="Input modalities (text, image, etc.)"
    )
    output: List[str] = Field(
        default_factory=list, description="Output modalities (text, etc.)"
    )


class ModelOptions(BaseModel):
    """Options specific to a model"""

    thinking: Optional[ThinkingConfig] = Field(
        default=None, description="Thinking configuration"
    )


class ModelConfig(BaseModel):
    """Configuration for a specific model"""

    name: str = Field(..., description="Display name of the model")
    modalities: Optional[Modalities] = Field(
        default=None, description="Input/output modalities"
    )
    options: Optional[ModelOptions] = Field(
        default=None, description="Model-specific options"
    )


class ProviderOptions(BaseModel):
    """Options for a provider"""

    base_url: str = Field(
        alias="baseURL", description="Base URL for the provider API"
    )
    api_key: str = Field(
        alias="apiKey", description="API key for the provider"
    )


class ProviderConfig(BaseModel):
    """Configuration for a provider"""

    name: str = Field(..., description="Display name of the provider")
    options: ProviderOptions = Field(description="Provider options")
    models: Dict[str, ModelConfig] = Field(
        ..., description="Available models for this provider"
    )


class ProviderContainer(BaseModel):
    """Container for all providers"""

    providers: Dict[str, ProviderConfig] = Field(alias="provider", default_factory=dict)


class CurrentChoice(BaseModel):
    provider: str = Field(description="Current provider")
    model: str = Field(description="Current model")


class MiniAgentConfig(BaseModel):
    """Root configuration for mini_agent.json"""

    provider: Dict[str, ProviderConfig] = Field(
        ..., description="Provider configurations"
    )
    current: CurrentChoice

    @classmethod
    def from_json_file(cls, file_path: str = "mini_agent.json") -> "MiniAgentConfig":
        """Load configuration from JSON file"""
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def get_provider(self, provider_id: str) -> ProviderConfig | None:
        """Get a specific provider by ID"""
        return self.provider.get(provider_id)

    def get_model(self, provider_id: str, model_id: str) -> Optional[ModelConfig]:
        """Get a specific model from a provider"""
        provider = self.get_provider(provider_id)
        if provider:
            return provider.models.get(model_id)
        return None

    def list_providers(self) -> List[str]:
        """List all available provider IDs"""
        return list(self.provider.keys())

    def list_models(self, provider_id: str) -> List[str]:
        """List all available model IDs for a provider"""
        provider = self.get_provider(provider_id)
        if provider:
            return list(provider.models.keys())
        return []


# Convenience function to load config
def load_config(file_path: str = "mini_agent.json") -> MiniAgentConfig:
    """Load and return the mini_agent configuration"""
    return MiniAgentConfig.from_json_file(file_path)


if __name__ == "__main__":
    # Example usage
    config = load_config()

    print("=" * 50)
    print("MiniAgent Configuration")
    print("=" * 50)

    for provider_id in config.list_providers():
        provider = config.get_provider(provider_id)
        if provider is None:
            continue
        print(f"\n📦 Provider: {provider_id}")
        print(f"   Name: {provider.name}")
        if provider.options:
            print(f"   Base URL: {provider.options.base_url}")
            print(f"   API Key: {'*' * 10 if provider.options.api_key else 'Not set'}")

        print(f"   Available models:")
        for model_id in config.list_models(provider_id):
            model = config.get_model(provider_id, model_id)
            if model is None:
                continue
            print(f"      • {model_id}: {model.name}")
            if model.options and model.options.thinking:
                print(
                    f"        Thinking: {model.options.thinking.type} ({model.options.thinking.budget_tokens} tokens)"
                )
    print(f"  Current:")
    print(config.current)
