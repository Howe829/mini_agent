from pydantic import BaseModel, Field
from typing import Type, override

from src.tools.tool_base import ToolBase, ToolReturnValue


class WeatherToolParams(BaseModel):
    city: str = Field(
        description="用以查询天气的城市名称，中文名称需要转成拼音，例如 上海-> Shanghai"
    )


class WeatherTool(ToolBase):
    name: str = "weather_tool"
    description: str = "查询指定城市的天气预报"
    params_class: Type[BaseModel] = WeatherToolParams

    @override
    def __call__(self, params: WeatherToolParams) -> ToolReturnValue:
        return ToolReturnValue(
            output=f"the weather in {params.city} is Coudy and the temperature is 15 celcius degrees",
            is_error=False,
        )
