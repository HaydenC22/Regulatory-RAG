from fastapi import APIRouter

from api.schemas import ChecklistRequest, ChecklistResponse
from services.agent.checklist import build_checklist

router = APIRouter(tags=["checklist"])


@router.post("/checklist", response_model=ChecklistResponse)
def checklist(request: ChecklistRequest) -> ChecklistResponse:
    return build_checklist(request.business_description)
