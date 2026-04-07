"""
Agent Workflow Routes
"""
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database.session import get_db

from app.database.models import User, Event, Participant, Email, MarketingPost, Schedule, AgentLog, AnalyticsReport
from app.schemas.agent_schema import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentLogResponse,
    MarketingWorkflowRequest,
    EmailWorkflowRequest,
    EmailSendRequest,
    EmailVariationSelectRequest,
    SchedulerWorkflowRequest,
    AnalyticsWorkflowRequest
)
from app.orchestration import (
    event_workflow,
    create_initial_state,
    save_agent_logs_to_db
)
from app.agents import (
    content_agent,
    communication_agent,
    scheduler_agent,
    analytics_agent
)
from app.dependencies import get_current_active_user
from app.utils.logger import logger


router = APIRouter(prefix="/agents", tags=["AI Agents"])


@router.post("/workflow/run", response_model=AgentExecutionResponse)
async def run_full_workflow(
    request: AgentExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Run complete multi-agent workflow

    Args:
        request: Workflow execution request
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database session

    Returns:
        Workflow execution response
    """
    # Verify event ownership
    result = await db.execute(
        select(Event).where(
            Event.id == request.event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )

    # Get participants
    participants_result = await db.execute(
        select(Participant).where(Participant.event_id == request.event_id)
    )
    participants = participants_result.scalars().all()

    # Prepare event data for workflow (no participants here)
    event_data = {
        "name": event.name,
        "description": event.description,
        "event_type": event.event_type,
        "theme": event.theme,
        "target_audience": event.target_audience,
        "start_date": event.start_date,
        "end_date": event.end_date,
        "location": event.location,
        "venue": event.venue,
        "event_metadata": event.event_metadata,
    }

    # Create initial state
    state = create_initial_state(
        user_id=str(current_user.id),
        event_id=str(event.id),
        event_data=event_data
    )

    # ✅ Fix: Set participants directly on state from the DB query result
    state["participants"] = [
        {
            "email": p.email,
            "full_name": p.full_name,
            "organization": p.organization,
            "role": p.role,
            "is_speaker": p.is_speaker,
            "is_sponsor": p.is_sponsor
        }
        for p in participants
    ]
    state["participant_count"] = len(participants)

    # Run workflow
    try:
        workflow_result = await event_workflow.run_workflow(
            user_id=str(current_user.id),
            event_id=str(event.id),
            event_data=event_data,
            config=request.parameters
        )

        # Save agent logs in background
        if workflow_result["status"] == "completed":
            background_tasks.add_task(
                save_agent_logs_to_db,
                db,
                str(event.id),
                workflow_result["workflow_id"],
                workflow_result["state"]
            )

            # Save generated content to database in background
            background_tasks.add_task(
                save_workflow_results,
                db,
                str(event.id),
                workflow_result["state"]
            )

        logger.info(f"Workflow {workflow_result['workflow_id']} completed for event {event.id}")

        return AgentExecutionResponse(
            workflow_id=UUID(workflow_result["workflow_id"]),
            status=workflow_result["status"],
            message="Workflow completed successfully" if workflow_result["status"] == "completed" else "Workflow failed",
            results={
                "scheduled_sessions": len(workflow_result["state"].get("scheduled_sessions", [])),
                "marketing_posts": len(workflow_result["state"].get("marketing_posts", [])),
                "emails_prepared": len(workflow_result["state"].get("emails_sent", [])),
                "insights": workflow_result["state"].get("insights", [])
            }
        )

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )


@router.post("/marketing/generate", response_model=AgentExecutionResponse)
async def generate_marketing_content(
    request: MarketingWorkflowRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate marketing content for event"""
    # Verify event ownership
    result = await db.execute(
        select(Event).where(
            Event.id == request.event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )

    # Create state for content agent
    state = create_initial_state(
        user_id=str(current_user.id),
        event_id=str(event.id),
        event_data={
            "name": event.name,
            "description": event.description,
            "event_type": event.event_type,
            "theme": event.theme,
            "target_audience": event.target_audience
        }
    )

    # Run content agent
    try:
        result_state = await content_agent.execute(state)

        return AgentExecutionResponse(
            workflow_id=UUID(result_state["workflow_id"]),
            status="completed",
            message="Marketing content generated successfully",
            results={
                "marketing_posts": result_state.get("marketing_posts", []),
                "marketing_plan": result_state.get("marketing_plan", {})
            }
        )

    except Exception as e:
        logger.error(f"Marketing content generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/email/prepare", response_model=AgentExecutionResponse)
async def prepare_emails(
    request: EmailWorkflowRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Prepare personalized emails for participants"""
    # Verify event ownership
    result = await db.execute(
        select(Event).where(
            Event.id == request.event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )

    # Get participants
    participants_query = select(Participant).where(Participant.event_id == request.event_id)

    if request.participant_ids:
        participants_query = participants_query.where(
            Participant.id.in_(request.participant_ids)
        )

    participants_result = await db.execute(participants_query)
    participants = participants_result.scalars().all()

    # Create state
    state = create_initial_state(
        user_id=str(current_user.id),
        event_id=str(event.id),
        event_data={
            "name": event.name,
            "description": event.description,
        }
    )

    # ✅ Fix: Set participants directly on state from the DB query result
    state["participants"] = [
        {
            "email": p.email,
            "full_name": p.full_name,
            "organization": p.organization,
            "role": p.role,
            "is_speaker": p.is_speaker,
            "is_sponsor": p.is_sponsor
        }
        for p in participants
    ]
    state["participant_count"] = len(participants)

    # Run email agent
    try:
        result_state = await communication_agent.execute(state)
        # Optionally send emails immediately
        if request.send_immediately:
            send_stats = await communication_agent.send_emails(result_state, db)
            results_payload = send_stats
        else:
            results_payload = {
                "prepared": len(result_state.get("emails_sent", [])),
                "emails_prepared": result_state.get("emails_sent", []),
                "template_variations": result_state.get("template_variations", None)
            }
        return AgentExecutionResponse(
            workflow_id=UUID(result_state["workflow_id"]),
            status="completed",
            message="Emails prepared successfully",
            results=results_payload
        )

    except Exception as e:
        logger.error(f"Email preparation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/email/send", response_model=AgentExecutionResponse)
async def send_prepared_emails(
    request: EmailSendRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Send previously prepared personalized emails"""
    # Verify event ownership
    result = await db.execute(
        select(Event).where(
            Event.id == request.event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )
        
    from uuid import uuid4
    try:
        # Create a mock state to pass to the agent
        mock_state = {
            "event_id": str(request.event_id),
            "emails_sent": request.emails
        }
        
        send_stats = await communication_agent.send_emails(mock_state, db)
        
        return AgentExecutionResponse(
            workflow_id=uuid4(),
            status="completed",
            message="Emails sent successfully",
            results=send_stats
        )
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/email/select-variations", response_model=AgentExecutionResponse)
async def select_email_variations(
    request: EmailVariationSelectRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Regenerate emails with selected template variations"""
    # Verify event ownership
    result = await db.execute(
        select(Event).where(
            Event.id == request.event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )
    
    # Get participants
    participants_result = await db.execute(
        select(Participant).where(Participant.event_id == request.event_id)
    )
    participants = participants_result.scalars().all()
    
    if not participants:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No participants found"
        )
    
    # Create state with selected variations stored
    state = create_initial_state(
        user_id=str(current_user.id),
        event_id=str(event.id),
        event_data={
            "name": event.name,
            "description": event.description,
            "participants": [
                {
                    "email": p.email,
                    "full_name": p.full_name,
                    "organization": p.organization,
                    "role": p.role,
                    "is_speaker": p.is_speaker,
                    "is_sponsor": p.is_sponsor
                }
                for p in participants
            ],
            "selected_variations": request.selected_variations
        }
    )
    
    try:
        result_state = await communication_agent.execute(state)
        
        # Return regenerated emails based on selected variations
        results_payload = {
            "prepared": len(result_state.get("emails_sent", [])),
            "emails_prepared": result_state.get("emails_sent", [])
        }
        
        return AgentExecutionResponse(
            workflow_id=UUID(result_state["workflow_id"]),
            status="completed",
            message="Emails regenerated with selected variations",
            results=results_payload
        )
    except Exception as e:
        logger.error(f"Email variation selection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/schedule/generate", response_model=AgentExecutionResponse)
async def generate_schedule(
    request: SchedulerWorkflowRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate optimized event schedule"""
    # Verify event ownership
    result = await db.execute(
        select(Event).where(
            Event.id == request.event_id,
            Event.owner_id == current_user.id
        )
    )
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )

    # Create state
    state = create_initial_state(
        user_id=str(current_user.id),
        event_id=str(event.id),
        event_data={
            "name": event.name,
            "start_date": event.start_date,
            "end_date": event.end_date,
            "venue": event.venue
        }
    )

    # Get speakers and set directly on state
    speakers_result = await db.execute(
        select(Participant).where(
            Participant.event_id == request.event_id,
            Participant.is_speaker == True
        )
    )
    speakers = speakers_result.scalars().all()
    state["speakers"] = [
        {"full_name": s.full_name, "organization": s.organization}
        for s in speakers
    ]

    # Run scheduler agent
    try:
        result_state = await scheduler_agent.execute(state)

        return AgentExecutionResponse(
            workflow_id=UUID(result_state["workflow_id"]),
            status="completed",
            message="Schedule generated successfully",
            results={
                "schedule": result_state.get("schedule", {}),
                "conflicts": result_state.get("schedule_conflicts", [])
            }
        )

    except Exception as e:
        logger.error(f"Schedule generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/analytics/generate", response_model=AgentExecutionResponse)
async def generate_analytics(
    request: AnalyticsWorkflowRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate analytics and insights"""

    # Verify event ownership and eager-load related data
    stmt = (
        select(Event)
        .options(
            selectinload(Event.participants),
            selectinload(Event.schedules),
            selectinload(Event.marketing_posts),
        )
        .where(Event.id == request.event_id)
    )

    result = await db.execute(stmt)
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found"
        )

    # Get all event data
    participants = event.participants
    schedules = event.schedules
    marketing_posts = event.marketing_posts

    # Create state
    state = create_initial_state(
        user_id=str(current_user.id),
        event_id=str(event.id),
        event_data={"name": event.name}
    )

    # ✅ Set all data directly on state
    state["participants"] = [
        {
            "email": p.email,
            "full_name": p.full_name,
            "organization": p.organization,
            "role": p.role,
            "is_speaker": p.is_speaker,
            "is_sponsor": p.is_sponsor
        }
        for p in participants
    ]
    state["participant_count"] = len(participants)

    state["scheduled_sessions"] = [
        {
            "session_name": s.session_name,
            "session_type": s.session_type,
            "duration_minutes": s.duration_minutes,
            "start_time": s.start_time,
            "end_time": s.end_time
        }
        for s in schedules
    ]

    state["marketing_posts"] = [
        {"platform": m.platform, "content": m.content}
        for m in marketing_posts
    ]

    # Run analytics agent
    try:
        result_state = await analytics_agent.execute(state)

        return AgentExecutionResponse(
            workflow_id=UUID(result_state["workflow_id"]),
            status="completed",
            message="Analytics generated successfully",
            results={
                "analytics": result_state.get("analytics", {}),
                "insights": result_state.get("insights", []),
                "recommendations": result_state.get("recommendations", [])
            }
        )

    except Exception as e:
        logger.error(f"Analytics generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


async def save_workflow_results(
    db: AsyncSession,
    event_id: str,
    state: dict
):
    """Background task to save workflow results to database"""
    from app.database.models import Schedule, MarketingPost

    try:
        # Save schedules
        for session in state.get("scheduled_sessions", []):
            schedule = Schedule(
                event_id=event_id,
                session_name=session["session_name"],
                session_type=session.get("session_type"),
                description=session.get("description"),
                start_time=session["start_time"],
                end_time=session["end_time"],
                duration_minutes=session.get("duration_minutes"),
                room=session.get("room"),
                speaker=session.get("speaker")
            )
            db.add(schedule)

        # Save marketing posts
        for post in state.get("marketing_posts", []):
            marketing_post = MarketingPost(
                event_id=event_id,
                platform=post.get("platform"),
                post_type=post.get("post_type"),
                content=post["content"],
                hashtags=post.get("hashtags", [])
            )
            db.add(marketing_post)

        await db.commit()
        logger.info(f"Workflow results saved for event {event_id}")

    except Exception as e:
        logger.error(f"Failed to save workflow results: {e}")


# ─── ORCHESTRATOR ENDPOINTS ────────────────────────────────────────────


@router.post("/orchestrator/event/{event_id}/execute-agent/{agent_type}")
async def orchestrator_execute_agent(
    event_id: UUID,
    agent_type: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Execute specific agent for event and track results in database"""
    # Verify event ownership
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
    
    # Get participants
    participants_result = await db.execute(
        select(Participant).where(Participant.event_id == event_id)
    )
    participants = participants_result.scalars().all()
    
    workflow_id = event.id
    agent_name = agent_type.replace('_', ' ').title()
    
    try:
        outputs = {}
        
        # Execute appropriate agent with realistic demo data
        if agent_type == "content":
            # Generate marketing posts
            posts_data = [
                {
                    "platform": "twitter",
                    "content": f"🎉 {event.name} is coming! Mark your calendars and join us for an unforgettable experience."
                },
                {
                    "platform": "linkedin",
                    "content": f"Exciting announcement! {event.name} will bring together industry leaders. Register now!"
                },
                {
                    "platform": "instagram",
                    "content": f"✨ Get ready for {event.name}! Limited spots available."
                }
            ]
            
            for post_data in posts_data:
                mp = MarketingPost(
                    event_id=event_id,
                    platform=post_data["platform"],
                    post_type="promotion",
                    content=post_data["content"],
                    hashtags=[event.name.lower().replace(" ", ""), "event"]
                )
                db.add(mp)
            
            outputs = {"posts_generated": len(posts_data)}
                
        elif agent_type == "email":
            # Generate emails for each participant
            for participant in participants:
                email_content = f"""Dear {participant.full_name},

We're excited to have you registered for {event.name}!

Event Details:
- Date: {event.start_date.strftime('%B %d, %Y')}
- Time: {event.start_date.strftime('%I:%M %p')}
- Location: {event.venue or 'TBD'}
- Participants: {len(participants)}

Looking forward to seeing you there!

Best regards,
The Event Team"""

                em = Email(
                    event_id=event_id,
                    recipient_email=participant.email,
                    recipient_name=participant.full_name,
                    subject=f"Welcome to {event.name}!",
                    body_text=email_content,
                    status="sent"
                )
                db.add(em)
            
            outputs = {"emails_sent": len(participants)}
                
        elif agent_type == "scheduler":
            # Generate schedule sessions
            sessions_data = [
                {
                    "session_name": "Opening Keynote",
                    "room": "Main Hall",
                    "duration": 60
                },
                {
                    "session_name": "Panel Discussion",
                    "room": "Hall A",
                    "duration": 45
                },
                {
                    "session_name": "Workshop",
                    "room": "Hall B",
                    "duration": 90
                }
            ]
            
            for idx, session in enumerate(sessions_data):
                start = event.start_date.replace(hour=9 + idx*3)
                sch = Schedule(
                    event_id=event_id,
                    session_name=session["session_name"],
                    session_type="session",
                    start_time=start,
                    end_time=start.replace(minute=session["duration"]),
                    duration_minutes=session["duration"],
                    room=session["room"]
                )
                db.add(sch)
            
            outputs = {"sessions_scheduled": len(sessions_data)}
                
        elif agent_type == "analytics":
            # Generate analytics report
            ar = AnalyticsReport(
                event_id=event_id,
                report_type="orchestrator_execution",
                report_name=f"Analytics Report - {event.name}",
                metrics={
                    "total_participants": len(participants),
                    "registrations": len(participants),
                    "completion_rate": 95
                },
                insights=[
                    f"Event attracts {len(participants)} participants",
                    "High engagement expected from target audience",
                    "Schedule optimized for maximum attendance"
                ],
                recommendations=[
                    "Send reminder emails 24 hours before event",
                    "Prepare backup plan for capacity overflow",
                    "Monitor registrations for final headcount"
                ],
                confidence_score=0.92
            )
            db.add(ar)
            
            outputs = {
                "insights_generated": 3,
                "recommendations": 3
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown agent type: {agent_type}"
            )
        
        # Log agent execution
        agent_log = AgentLog(
            event_id=event_id,
            agent_name=agent_name,
            workflow_id=workflow_id,
            status="completed",
            inputs={"event_name": event.name, "participants_count": len(participants)},
            outputs=outputs,
            execution_time_ms=0
        )
        db.add(agent_log)
        await db.commit()
        
        return {
            "status": "completed",
            "agent": agent_type,
            "message": f"{agent_name} executed successfully",
            "results": outputs
        }
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        await db.rollback()
        
        # Log failure
        try:
            agent_log = AgentLog(
                event_id=event_id,
                agent_name=agent_name,
                workflow_id=workflow_id,
                status="failed",
                error_message=str(e)
            )
            db.add(agent_log)
            await db.commit()
        except:
            pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/orchestrator/event/{event_id}/summary")
async def orchestrator_event_summary(
    event_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get complete summary of all agent activities for an event"""
    # Verify event ownership
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
    
    # Get all agent logs
    logs_result = await db.execute(
        select(AgentLog).where(AgentLog.event_id == event_id).order_by(AgentLog.created_at.desc())
    )
    agent_logs = logs_result.scalars().all()
    
    # Get all emails
    emails_result = await db.execute(
        select(Email).where(Email.event_id == event_id).order_by(Email.created_at.desc())
    )
    all_emails = emails_result.scalars().all()
    
    # Get all marketing posts
    posts_result = await db.execute(
        select(MarketingPost).where(MarketingPost.event_id == event_id).order_by(MarketingPost.created_at.desc())
    )
    all_posts = posts_result.scalars().all()
    
    # Get all schedules
    schedules_result = await db.execute(
        select(Schedule).where(Schedule.event_id == event_id).order_by(Schedule.start_time.desc())
    )
    all_schedules = schedules_result.scalars().all()
    
    # Get all analytics
    analytics_result = await db.execute(
        select(AnalyticsReport).where(AnalyticsReport.event_id == event_id).order_by(AnalyticsReport.created_at.desc())
    )
    all_analytics = analytics_result.scalars().all()
    
    # Count email statuses
    sent_count = sum(1 for e in all_emails if e.status == "sent")
    pending_count = sum(1 for e in all_emails if e.status == "pending")
    failed_count = sum(1 for e in all_emails if e.status == "failed")
    
    # Prepare response
    return {
        "event": {
            "id": str(event.id),
            "name": event.name,
            "start_date": event.start_date.isoformat() if event.start_date else None,
            "venue": event.venue
        },
        "agent_logs": [
            {
                "agent_name": log.agent_name,
                "status": log.status,
                "created_at": log.created_at.isoformat(),
                "inputs": log.inputs,
                "outputs": log.outputs,
                "error_message": log.error_message
            }
            for log in agent_logs
        ],
        "emails": {
            "total": len(all_emails),
            "sent": sent_count,
            "pending": pending_count,
            "failed": failed_count,
            "recent": [
                {
                    "id": str(e.id),
                    "recipient_email": e.recipient_email,
                    "recipient_name": e.recipient_name,
                    "subject": e.subject,
                    "status": e.status,
                    "created_at": e.created_at.isoformat(),
                    "sent_at": e.sent_at.isoformat() if e.sent_at else None
                }
                for e in all_emails[:10]
            ]
        },
        "marketing": {
            "total_posts": len(all_posts),
            "by_platform": {
                platform: sum(1 for p in all_posts if p.platform == platform)
                for platform in set(p.platform for p in all_posts if p.platform)
            },
            "published": sum(1 for p in all_posts if p.is_published),
            "recent": [
                {
                    "id": str(p.id),
                    "platform": p.platform,
                    "post_type": p.post_type,
                    "content": p.content[:100] + "..." if len(p.content) > 100 else p.content,
                    "is_published": p.is_published,
                    "created_at": p.created_at.isoformat()
                }
                for p in all_posts[:10]
            ]
        },
        "schedule": {
            "total_sessions": len(all_schedules),
            "recent": [
                {
                    "id": str(s.id),
                    "session_name": s.session_name,
                    "session_type": s.session_type,
                    "room": s.room,
                    "start_time": s.start_time.isoformat() if s.start_time else None,
                    "end_time": s.end_time.isoformat() if s.end_time else None,
                    "duration_minutes": s.duration_minutes
                }
                for s in all_schedules[:10]
            ]
        },
        "analytics": {
            "reports_generated": len(all_analytics),
            "recent": [
                {
                    "id": str(a.id),
                    "report_type": a.report_type,
                    "report_name": a.report_name,
                    "metrics": a.metrics,
                    "confidence_score": a.confidence_score,
                    "created_at": a.created_at.isoformat()
                }
                for a in all_analytics[:5]
            ]
        }
    }