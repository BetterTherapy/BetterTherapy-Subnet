from sqlalchemy import update
from .session import dataset_session, session
from .models import BlacklistedMiners, BasePromptResponse, Request, MinerResponse
from datetime import datetime, timedelta, timezone
import typing
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.dialects.sqlite import insert


# get request with greater than or equal to 24 hours old
@session
def get_ready_requests(session: Session, hours: int = 24) -> typing.List[Request]:
    """Get requests that are older than the specified number of hours."""

    threshold = datetime.now(timezone.utc) - timedelta(minutes=16)
    # threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
    # attach responses to the requests
    return (
        session.query(Request)
        .filter(Request.created_at < threshold)
        .options(selectinload(Request.responses))
        .all()
    )


@session
def get_blacklisted_miners_hotkeys(session: Session):
    """
    Fetch all blacklisted miners from the database.
    """
    return (
        session.query(BlacklistedMiners.hotkey)
        .where(BlacklistedMiners.blacklist_count > 1)
        .all()
    )


@session
def add_or_update_blacklisted_miner(
    session: Session, miner_id: int, hotkey: str, coldkey: str, reason: str
):
    """
    Add or update a blacklisted miner in the database.
    If the hotkey already exists, it will be updated with the new miner_id.
    """
    now = datetime.utcnow().isoformat()

    ups_stmt = insert(BlacklistedMiners).values(
        miner_id=miner_id,
        hotkey=hotkey,
        updated_at=now,
        coldkey=coldkey,
        reason=reason,
    )
    query = ups_stmt.on_conflict_do_update(
        index_elements=["hotkey"],
        set_=dict(
            miner_id=ups_stmt.excluded.miner_id,
            updated_at=now,
            coldkey=ups_stmt.excluded.coldkey,
            blacklist_count=BlacklistedMiners.blacklist_count + 1,
        ),
    )
    session.execute(query)
    session.commit()


@session
def add_request(
    session: Session, name: str, openai_batch_id: str, prompt: str, base_response: str
) -> Request:
    """Add a new request to the database."""
    new_request = Request(
        name=name,
        openai_batch_id=openai_batch_id,
        prompt=prompt,
        base_response=base_response,
    )
    session.add(new_request)
    session.commit()
    session.refresh(new_request)
    return new_request


@session
def add_bulk_responses(session: Session, responses: typing.List[MinerResponse]) -> None:
    """Add multiple responses to a request."""
    session.bulk_save_objects(responses)
    session.commit()


@session
def delete_requests(session: Session, request_ids: typing.List[int]) -> None:
    """Delete requests by their IDs."""
    session.query(Request).filter(Request.id.in_(request_ids)).delete(
        synchronize_session=False
    )
    session.commit()


# Dataset DB Queries
@dataset_session
def get_base_prompt_response(session: Session) -> typing.Tuple[BasePromptResponse, int]:
    """Get a base prompt response by prompt."""
    not_used_count = (
        session.query(BasePromptResponse)
        .filter(BasePromptResponse.is_used == False)  # noqa: E712
        .count()
    )
    single_item = (
        session.query(BasePromptResponse)
        .filter(BasePromptResponse.is_used == False)  # noqa: E712
        .first()
    )
    return single_item, not_used_count


@dataset_session
def update_base_prompt_response(
    session: Session, base_prompt_response_id: int
) -> BasePromptResponse:
    """Add a base prompt response to the database."""
    session.execute(
        update(BasePromptResponse)
        .where(BasePromptResponse.id == base_prompt_response_id)
        .values(is_used=True)
    )
    session.commit()
