from __future__ import annotations

from property_guardian_ai.agents.base import BaseAgent
from property_guardian_ai.models.schemas import PropertyInput, PropertyProfile


class PropertyAgent(BaseAgent):
    name = "property-agent"

    def run(self, property_input: PropertyInput) -> PropertyProfile:
        property_age = max(0, property_input.as_of.year - property_input.year_built)
        roof_age_estimated = property_input.roof_age is None
        roof_age = property_age // 2 if roof_age_estimated else property_input.roof_age

        return PropertyProfile(
            property_id=property_input.property_id,
            address=property_input.address,
            latitude=property_input.latitude,
            longitude=property_input.longitude,
            year_built=property_input.year_built,
            property_age=property_age,
            roof_age=roof_age,
            roof_age_estimated=roof_age_estimated,
            home_value=property_input.home_value,
            square_footage=property_input.square_footage,
            as_of=property_input.as_of,
        )
