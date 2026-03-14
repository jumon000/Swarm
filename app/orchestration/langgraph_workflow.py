"""
LangGraph Workflow Orchestration
Multi-agent workflow using LangGraph
"""
from typing import Dict, Any, List
from uuid import uuid4
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


from app.orchestration.state_schema import (
    AgentState,
    WorkflowConfig,
    create_initial_state,
    get_default_config
)
from app.agents import (
    content_agent,
    communication_agent,
    scheduler_agent,
    analytics_agent
)
from app.database.models import Event, Participant, AgentLog
from app.memory.vector_store import vector_store
from app.utils.logger import logger


class EventWorkflow:
    """LangGraph workflow for event logistics automation"""
    
    def __init__(self):
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph state graph"""
        
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("load_context", self.load_context_node)
        workflow.add_node("scheduler", self.scheduler_node)
        workflow.add_node("marketing", self.marketing_node)
        workflow.add_node("email", self.email_node)
        workflow.add_node("analytics_agent", self.analytics_node)
        workflow.add_node("save_results", self.save_results_node)
        
        # Define edges (workflow flow)
        workflow.set_entry_point("load_context")
        
        workflow.add_edge("load_context", "scheduler")
        workflow.add_edge("scheduler", "marketing")
        workflow.add_edge("marketing", "email")
        workflow.add_edge("email", "analytics_agent")
        workflow.add_edge("analytics_agent", "save_results")
        workflow.add_edge("save_results", END)
        
        # Compile graph
        self.graph = workflow.compile()
        
        logger.info("LangGraph workflow built successfully")
    
    async def load_context_node(self, state: AgentState) -> AgentState:
        """
        Load context and event data
        
        Args:
            state: Current state
            
        Returns:
            Updated state with context
        """
        logger.info("Loading context...")
        
        try:
            # Retrieve event memories from vector store (handle errors gracefully)
            try:
                memories = vector_store.get_event_context(
                    event_id=state["event_id"],
                    k=3
                )
            except Exception as e:
                logger.warning(f"Vector store retrieval failed: {e}")
                memories = []
            
            # Retrieve user preferences (handle errors gracefully)
            try:
                user_prefs = vector_store.get_user_preferences(
                    user_id=state["user_id"]
                )
            except Exception as e:
                logger.warning(f"User preferences retrieval failed: {e}")
                user_prefs = []
            
            # IMPORTANT FIX: Load participants from state
            participants = state.get("participants", [])
            speakers = []
            sponsors = []
            
            # Segment participants by role
            for p in participants:
                if p.get("is_speaker"):
                    speakers.append(p)
                if p.get("is_sponsor"):
                    sponsors.append(p)
            
            logger.info(f"Loaded {len(participants)} participants ({len(speakers)} speakers, {len(sponsors)} sponsors)")
            
            # Update state with retrieved context and participants
            return {
                **state,
                "participants": participants,
                "participant_count": len(participants),
                "speakers": speakers,
                "sponsors": sponsors,
                "retrieved_memories": memories,
                "context": {
                    "memories": memories,
                    "user_preferences": user_prefs,
                    "loaded_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            errors = state.get("errors", [])
            errors.append(f"Context loading failed: {str(e)}")
            return {**state, "errors": errors}
    
    async def scheduler_node(self, state: AgentState) -> AgentState:
        """
        Execute scheduler agent
        
        Args:
            state: Current state
            
        Returns:
            Updated state with schedule
        """
        logger.info("Executing scheduler agent...")
        return await scheduler_agent.execute(state)
    
    async def marketing_node(self, state: AgentState) -> AgentState:
        """
        Execute marketing/content agent
        
        Args:
            state: Current state
            
        Returns:
            Updated state with marketing content
        """
        logger.info("Executing marketing agent...")
        return await content_agent.execute(state)
    
    async def email_node(self, state: AgentState) -> AgentState:
        """
        Execute email/communication agent
        
        Args:
            state: Current state
            
        Returns:
            Updated state with email data
        """
        logger.info("Executing email agent...")
        return await communication_agent.execute(state)
    
    async def analytics_node(self, state: AgentState) -> AgentState:
        """
        Execute analytics agent
        
        Args:
            state: Current state
            
        Returns:
            Updated state with analytics
        """
        logger.info("Executing analytics agent...")
        return await analytics_agent.execute(state)
    
    async def save_results_node(self, state: AgentState) -> AgentState:
        """
        Save results to vector memory
        
        Args:
            state: Current state
            
        Returns:
            Final state
        """
        logger.info("Saving results to memory...")
        
        try:
            # Store event summary in vector memory
            event_summary = self._create_event_summary(state)
            
            vector_store.add_event_memory(
                event_id=state["event_id"],
                content=event_summary,
                metadata={
                    "event_type": state.get("event_type"),
                    "participant_count": state.get("participant_count"),
                    "workflow_id": state.get("workflow_id")
                }
            )
            
            logger.info("Results saved to vector memory")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        return state
    
    def _create_event_summary(self, state: AgentState) -> str:
        """Create text summary of event for vector storage"""
        
        summary_parts = [
            f"Event: {state.get('event_name')}",
            f"Type: {state.get('event_type')}",
            f"Participants: {state.get('participant_count')}",
            f"Sessions Scheduled: {len(state.get('scheduled_sessions', []))}",
            f"Marketing Posts: {len(state.get('marketing_posts', []))}",
            f"Emails Prepared: {len(state.get('emails_sent', []))}",
        ]
        
        # Add insights
        insights = state.get("insights", [])
        if insights:
            summary_parts.append(f"Key Insights: {', '.join(insights[:3])}")
        
        return "\n".join(summary_parts)
    
    async def run_workflow(
        self,
        user_id: str,
        event_id: str,
        event_data: Dict[str, Any],
        config: WorkflowConfig = None
    ) -> Dict[str, Any]:
        """
        Run the complete workflow
        
        Args:
            user_id: User ID
            event_id: Event ID
            event_data: Event information
            config: Workflow configuration
            
        Returns:
            Final workflow state
        """
        workflow_id = str(uuid4())
        logger.info(f"Starting workflow {workflow_id} for event {event_id}")
        
        # Create initial state
        # Create initial state
        initial_state = create_initial_state(user_id, event_id, event_data)
        initial_state["workflow_id"] = workflow_id

        # CRITICAL FIX: Ensure participants are in the initial state
        if "participants" in event_data:
            initial_state["participants"] = event_data["participants"]
            initial_state["participant_count"] = len(event_data["participants"])
            logger.info(f"Loaded {len(event_data['participants'])} participants into initial state")
        else:
            logger.warning("No participants provided in event_data!")
        
        # Use default config if not provided
        if config is None:
            config = get_default_config()
        
        try:
            # Execute workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            logger.info(f"Workflow {workflow_id} completed successfully")
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "state": final_state,
                "errors": final_state.get("errors", []),
                "warnings": final_state.get("warnings", [])
            }
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "state": initial_state
            }
    
    async def run_single_agent(
        self,
        agent_name: str,
        state: AgentState
    ) -> AgentState:
        """
        Run a single agent
        
        Args:
            agent_name: Name of agent to run
            state: Current state
            
        Returns:
            Updated state
        """
        agents = {
            "scheduler": scheduler_agent,
            "marketing": content_agent,
            "email": communication_agent,
            "analytics": analytics_agent
        }
        
        if agent_name not in agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = agents[agent_name]
        return await agent.execute(state)


async def save_agent_logs_to_db(
    db: AsyncSession,
    event_id: str,
    workflow_id: str,
    state: AgentState
) -> None:
    """
    Save agent execution logs to database
    
    Args:
        db: Database session
        event_id: Event ID
        workflow_id: Workflow ID
        state: Final workflow state
    """
    try:
        for agent_name, output in state.get("agent_outputs", {}).items():
            log = AgentLog(
                event_id=event_id,
                workflow_id=workflow_id,
                agent_name=agent_name,
                status="completed",
                inputs={"state_keys": list(state.keys())},
                outputs=output,
                execution_time_ms=0  # Would be calculated during execution
            )
            
            db.add(log)
        
        await db.commit()
        logger.info(f"Saved agent logs for workflow {workflow_id}")
        
    except Exception as e:
        logger.error(f"Failed to save agent logs: {e}")
        await db.rollback()


# Create singleton workflow instance
event_workflow = EventWorkflow()