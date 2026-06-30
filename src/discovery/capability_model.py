from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Affordance:
    name: str
    endpoint_url: str
    semantic_type: str
    op_type: str  # "invokeaction" | "readproperty" | "writeproperty"
    schema: dict = field(default_factory=dict)


@dataclass
class Artifact:
    name: str
    kg_uri: str
    actions: list[Affordance] = field(default_factory=list)
    properties: list[Affordance] = field(default_factory=list)
    location: dict = field(default_factory=dict)
    current_state: dict = field(default_factory=dict)


@dataclass
class CapabilityModel:
    goal: str
    artifacts: dict[str, Artifact] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        """Format capability model as markdown for LLM injection."""
        lines = []
        for name, artifact in self.artifacts.items():
            lines.append(f"## {name}")
            lines.append(f"URI: `{artifact.kg_uri}`")

            if artifact.actions:
                lines.append("\n**Actions:**")
                for action in artifact.actions:
                    lines.append(
                        f"  - **{action.name}** [{action.semantic_type}]"
                    )
                    lines.append(f"    POST `{action.endpoint_url}`")
                    if action.schema:
                        params_str = ", ".join(
                            f"{k}: {v.get('type', 'any')}"
                            for k, v in action.schema.get("properties", {}).items()
                        )
                        if params_str:
                            lines.append(f"    Input: {{{params_str}}}")

            if artifact.properties:
                lines.append("\n**Properties:**")
                for prop in artifact.properties:
                    lines.append(
                        f"  - **{prop.name}** [{prop.semantic_type}]"
                    )
                    method = "PUT" if "writeproperty" in prop.op_type else "GET"
                    lines.append(f"    {method} `{prop.endpoint_url}`")
                    if prop.schema:
                        schema_type = prop.schema.get("type", "unknown")
                        lines.append(f"    Type: {schema_type}")

            if artifact.location:
                lines.append("\n**Location:**")
                if "origin" in artifact.location:
                    origin = artifact.location["origin"]
                    lines.append(
                        f"  Origin: ({origin.get('x')}, {origin.get('y')}, {origin.get('z')})"
                    )
                if "output_area" in artifact.location:
                    output = artifact.location["output_area"]
                    lines.append(
                        f"  Output Area: ({output.get('x')}, {output.get('y')}, {output.get('z')})"
                    )
                if "input_area" in artifact.location:
                    input_area = artifact.location["input_area"]
                    lines.append(
                        f"  Input Area: ({input_area.get('x')}, {input_area.get('y')}, {input_area.get('z')})"
                    )

            if artifact.current_state:
                lines.append("\n**Current State:**")
                for prop_name, value in artifact.current_state.items():
                    lines.append(f"  - {prop_name}: {value}")

            lines.append("")

        return "\n".join(lines)
