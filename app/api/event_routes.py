"""
Event Management Routes
"""
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database.session import get_db
from app.database.models import User, Event, EventStatus
from app.schemas.event_schema import (
    EventCreate,
    EventUpdate,
    EventResponse,
    ScheduleResponse,
    MarketingPostResponse
)
from app.dependencies import get_current_active_user
from app.utils.logger import logger


router = APIRouter(prefix="/events", tags=["Events"])


@router.post("", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def create_event(
    event_data: EventCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new event
    
    Args:
        event_data: Event creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created event
    """
    # Validate dates
    if event_data.end_date <= event_data.start_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="End date must be after start date"
        )
    
    # Create event
    new_event = Event(
        owner_id=current_user.id,
        name=event_data.name,
        description=event_data.description,
        event_type=event_data.event_type,
        theme=event_data.theme,
        target_audience=event_data.target_audience,
        start_date=event_data.start_date,
        end_date=event_data.end_date,
        location=event_data.location,
        venue=event_data.venue,
        max_participants=event_data.max_participants,
        metadata=event_data.event_metadata or {}
    )
    
    db.add(new_event)
    await db.commit()
    await db.refresh(new_event)
    
    logger.info(f"Event created: {new_event.name} by user {current_user.username}")
    
    return new_event


@router.get("", response_model=List[EventResponse])
async def list_user_events(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    status_filter: EventStatus = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all events for current user
    
    Args:
        current_user: Current authenticated user
        db: Database session
        status_filter: Optional status filter
        limit: Maximum number of results
        offset: Offset for pagination
        
    Returns:
        List of events
    """
    query = select(Event).where(Event.owner_id == current_user.id)
    
    if status_filter:
        query = query.where(Event.status == status_filter)
    
    query = query.limit(limit).offset(offset).order_by(Event.created_at.desc())
    
    result = await db.execute(query)
    events = result.scalars().all()
    
    return events


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get event by ID
    
    Args:
        event_id: Event ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Event details
    """
    result = await db.execute(
        select(Event).where(
            Event.id == event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )
    
    return event


@router.put("/{event_id}", response_model=EventResponse)
async def update_event(
    event_id: UUID,
    event_data: EventUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update event
    
    Args:
        event_id: Event ID
        event_data: Updated event data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated event
    """
    result = await db.execute(
        select(Event).where(
            Event.id == event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )
    
    # Update fields
    update_data = event_data.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(event, field, value)
    
    await db.commit()
    await db.refresh(event)
    
    logger.info(f"Event updated: {event.name}")
    
    return event


@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event(
    event_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete event
    
    Args:
        event_id: Event ID
        current_user: Current authenticated user
        db: Database session
    """
    result = await db.execute(
        select(Event).where(
            Event.id == event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )
    
    await db.delete(event)
    await db.commit()
    
    logger.info(f"Event deleted: {event.name}")


@router.get("/{event_id}/schedule", response_model=List[ScheduleResponse])
async def get_event_schedule(
    event_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get event schedule
    
    Args:
        event_id: Event ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        List of scheduled sessions
    """
    # Verify event ownership
    result = await db.execute(
        select(Event)
        .options(selectinload(Event.schedules))  # ✅ Eager load!
        .where(
            Event.id == event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )

    return event.schedules  # ✅ Already loaded, no error


@router.get("/{event_id}/marketing", response_model=List[MarketingPostResponse])
async def get_event_marketing(
    event_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get event marketing posts
    
    Args:
        event_id: Event ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        List of marketing posts
    """
    # FIXED: Eager load marketing_posts relationship
    result = await db.execute(
        select(Event)
        .options(selectinload(Event.marketing_posts))  # ✅ Added this!
        .where(
            Event.id == event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )
    
    return event.marketing_posts  # ✅ Now safely loaded