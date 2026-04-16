import logging
import json
from ..models import AnalysisSession, ChatMessage

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Centralized utility for building structured contexts for Gemini agents.
    Enforces the split between STATIC (cacheable) and DYNAMIC (live) context.
    """

    @staticmethod
    def build_static_context(system_instruction: str, schema_text: str, tools_json: str) -> str:
        """
        Combines elements that are static for the duration of a dataset session.
        This block is the primary candidate for Context Caching.
        """
        return f"""
        {system_instruction}

        ### DATASET SCHEMA
        {schema_text}

        ### AVAILABLE TOOLS
        {tools_json}
        """

    @staticmethod
    def _get_recent_messages(session, limit=5):
        """
        Helper to get meaningful conversation history (Elite v3.3).
        - Hard-capped at last 5 interactions.
        - Filters for turns with analytical substance.
        """
        # Fetch the most recent turns
        messages = ChatMessage.objects.filter(session=session).order_by('-created_at')
        
        meaningful_turns = []
        for msg in messages:
            if len(meaningful_turns) >= limit:
                break
                
            # Filter criteria:
            # 1. User messages are ALWAYS included if they led to analysis
            # 2. AI messages are included if they contain tool artifacts or were successfully metadata-tagged
            has_artifacts = msg.metadata and (msg.metadata.get("id") or msg.metadata.get("tool_used"))
            
            if msg.message_type == 'user' or has_artifacts:
                meaningful_turns.append(msg)

        formatted = []
        for msg in reversed(meaningful_turns):
            role = "User" if msg.message_type == 'user' else "AI"
            # Strip metadata from prompt injection to keep it clean
            content = msg.content.split("\n\n[METADATA]")[0].strip()
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)

    @staticmethod
    def build_planner_context(session: AnalysisSession, query: str, include_summary: bool = True) -> str:
        """
        Builds the DYNAMIC context for the Analysis Planner using the Authority Layer.
        """
        from .context_resolver import ContextResolver
        
        resolved_state = ContextResolver.resolve_state(session)
        
        # Priority: If not a follow-up, we might still include summary if include_summary is True,
        # but the Resolver handles the unification.
        context_block = ContextResolver.format_resolved_state(resolved_state) if include_summary else ""
        
        recent_history = ContextBuilder._get_recent_messages(session, limit=2)
        
        return f"""{context_block}
        
        ### RECENT CONVERSATION
        {recent_history}

        ### CURRENT USER QUERY
        {query}
        """

    @staticmethod
    def build_interpreter_context(session: AnalysisSession, tool_results: str) -> str:
        """
        Builds the DYNAMIC context for the Interpretation Agent.
        Focuses on results, findings, and coherence with past state.
        """
        summary = session.memory_summary or "No previous analysis history."
        working_memory = json.dumps(session.working_memory or {}, indent=2)
        recent_history = ContextBuilder._get_recent_messages(session, limit=3)

        return f"""
        ### LONG-TERM MEMORY (SUMMARY)
        {summary}

        ### WORKING MEMORY (ACTIVE STATE)
        {working_memory}

        ### RECENT CONVERSATION
        {recent_history}

        ### NEW TOOL OUTPUT (FOR INTERPRETATION)
        {tool_results}
        """
