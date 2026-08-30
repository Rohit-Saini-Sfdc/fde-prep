from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List

class CompanyRecord(BaseModel):
    """Company details extracted from nested payloads."""
    model_config = ConfigDict(populate_by_name=True)
    
    name: str
    catch_phrase: Optional[str] = Field(default=None, alias="catchPhrase")
    bs: Optional[str] = None

class UserRecord(BaseModel):
    """User schema with strict validation and alias mapping."""
    model_config = ConfigDict(populate_by_name=True)
    
    id: int
    name: str
    email: str
    company_name: str = Field(validation_alias="company_name")
    city: Optional[str] = None

    @classmethod
    def from_nested_json(cls, raw_json: dict) -> "UserRecord":
        """Flatten nested API response into valid Pydantic model instance."""
        flat = {
            "id": raw_json["id"],
            "name": raw_json["name"],
            "email": raw_json["email"],
            "company_name": raw_json.get("company", {}).get("name", "N/A"),
            "city": raw_json.get("address", {}).get("city", "N/A"),
        }
        return cls.model_validate(flat)
