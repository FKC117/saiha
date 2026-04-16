import logging
import json
import re
from datetime import timedelta
from django.utils import timezone
from ..models import AnalysisSession, ChatMessage, AnalysisResult

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages the structured rolling memory and working memory for the agent.
    Ensures context is compressed and relevant without token explosion.
    Elite Mode (v3.1) features: hard resets, decay, and structured metadata.
    """
    
    SUMMARY_INTERVAL = 5
    REBUILD_THRESHOLD = 20 # Deep rebuild every 20 messages to prevent drift
    DECAY_TIMEOUT_MINS = 30 # Working memory decay after 30 mins idle
    
    # Elite v3.2 Hard Limits
    SUMMARY_MAX_CHARS = 1000
    ANALYSIS_CHAIN_MAX = 5
    ACTIVE_COLUMNS_MAX = 5
    
    @staticmethod
    def should_update_summary(session: AnalysisSession, event: str = None) -> bool:
        """
        Determines if a summary update is warranted.
        """
        if event in ["tool_executed", "analysis_completed"]:
            return True
            
        message_count = ChatMessage.objects.filter(session=session).count()
        if (message_count - session.last_summary_at_message) >= MemoryManager.SUMMARY_INTERVAL:
            return True
            
        return False

    @staticmethod
    def decay_stale_state(session: AnalysisSession, force: bool = False):
        """
        Wipes volatile working memory if the session has been idle for too long,
        OR if a forced reset is triggered by an intent override.
        Preserves long-term memory (summary).
        """
        idle_duration = timezone.now() - session.last_activity
        if force or idle_duration > timedelta(minutes=MemoryManager.DECAY_TIMEOUT_MINS):
            logger.info(f"Decaying/Resetting volatile working memory for session {session.id} (Force: {force})")
            # Clear volatile intent/state
            volatile_fields = ["active_columns", "last_tool", "last_result_type", "last_target_column"]
            wm = session.working_memory or {}
            for field in volatile_fields:
                if field in wm:
                    wm[field] = None
            
            session.working_memory = wm
            # Also clear the recent analysis chain on forced reset (override)
            if force:
                session.analysis_chain = []
                
            session.save(update_fields=['working_memory', 'analysis_chain'])
            return True
        return False

    @staticmethod
    def update_summary(session: AnalysisSession, llm_service, force_rebuild=False):
        """
        Updates the session's memory_summary using the LLM.
        Enforces a strict structured format.
        """
        try:
            message_count = ChatMessage.objects.filter(session=session).count()
            
            # Check for deep rebuild trigger
            if message_count >= MemoryManager.REBUILD_THRESHOLD and not force_rebuild:
                # Every 20 messages, we do a hard reset to flush compression artifacts
                if message_count % MemoryManager.REBUILD_THRESHOLD == 0:
                    return MemoryManager.rebuild_memory_from_scratch(session, llm_service)

            # 1. Fetch recent history for summarization
            history = ChatMessage.objects.filter(session=session).order_by('created_at')
            history_text = ""
            for msg in history:
                role = "User" if msg.message_type == 'user' else "AI"
                history_text += f"{role}: {msg.content}\n"

            # 2. Build the summarization prompt
            prompt = f"""
            You are a memory management sub-process for a data analysis AI. 
            Your goal is to compress the conversation history into a STRUCTURED SUMMARY.

            CURRENT SUMMARY:
            {session.memory_summary or "No summary yet."}

            NEW CONVERSATION LOG:
            {history_text}

            TASK:
            Update the summary based on the new logs. 
            MANDATORY STRUCTURE (Strictly follow this):
            DATASET: <name/type>
            COLUMNS_USED: [<list of columns discussed/analyzed>]
            ANALYSES_RUN: [<list of tools/tests executed>]
            KEY_FINDINGS: <concise bullet points of primary statistical insights>
            NEXT_STEPS: <1-2 suggested next logical analysis steps>

            TONE: Concise, technical, and objective.
            """

            updated_summary = llm_service.generate_content(prompt)
            
            # 3. Validation Guardrail (v3.2: 1-Retry Strategy)
            retry_count = 0
            while retry_count < 1:
                # Reject if summary doesn't contain Key Findings or is too short
                if "KEY_FINDINGS:" in updated_summary and len(updated_summary) > 50:
                    break # Success
                
                logger.warning(f"Summary validation failed. Retrying... (Attempt {retry_count + 1})")
                retry_prompt = f"STRICT QUALITY GUARD: The previous summary was rejected for missing KEY_FINDINGS or being too brief. RE-GENERATE NOW WITH STRICT ADHERENCE TO THE TEMPLATE.\n\n{prompt}"
                updated_summary = llm_service.generate_content(retry_prompt)
                retry_count += 1

            # Final check before saving
            if "KEY_FINDINGS:" not in updated_summary or len(updated_summary) < 50:
                logger.error(f"Failed to generate valid summary after retry. FREEZING old summary for session {session.id}")
                return False

            session.memory_summary = updated_summary.strip()[:MemoryManager.SUMMARY_MAX_CHARS]
            session.last_summary_at_message = message_count
            session.save(update_fields=['memory_summary', 'last_summary_at_message'])
            
            logger.info(f"Updated memory summary for session {session.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update memory summary: {e}")
            return False

    @staticmethod
    def rebuild_memory_from_scratch(session: AnalysisSession, llm_service):
        """
        Deep rebuild to prevent LLM compression drift. 
        Uses structured snapshots of the last 15-20 meaningful interactions.
        """
        logger.info(f"Performing deep memory rebuild for session {session.id}")
        
        # Pull last 15 meaningful tool results
        results = AnalysisResult.objects.filter(session=session).order_by('-completed_at')[:15]
        snapshots = []
        for r in reversed(results):
            snapshots.append({
                "query": r.query or "N/A",
                "tool": r.tool_used,
                "findings": (r.ai_interpretation or "")[:200], # Trucate for efficiency
                "status": r.status
            })

        history_json = json.dumps(snapshots, indent=2)
        
        prompt = f"""
        RECONSTRUCTIVE SUMMARY TASK:
        The existing session summary is being discarded to prevent "compression drift".
        Rebuild a fresh, high-precision summary based ONLY on these technical snapshots of past analysis results:

        SNAPSHOTS:
        {history_json}

        MANDATORY STRUCTURE:
        DATASET: <dataset name>
        COLUMNS_USED: [<list>]
        ANALYSES_RUN: [<list>]
        KEY_FINDINGS: <detailed technical insights>
        NEXT_STEPS: <suggested continuations>
        """

        updated_summary = llm_service.generate_content(prompt)
        session.memory_summary = updated_summary.strip()
        session.save(update_fields=['memory_summary'])
        return True

    @staticmethod
    def update_working_memory(session: AnalysisSession, metadata: dict = None):
        """
        Updates the micro-state (Working Memory) and the analysis chain.
        Wipes oldest steps from chain to keep it at 5.
        """
        wm = session.working_memory or {}
        chain = session.analysis_chain or []

        # Update fields with provided metadata
        for key in ["active_columns", "last_tool", "last_result_type", "last_target_column"]:
            if key in metadata:
                wm[key] = metadata[key]
        
        # Update analysis chain
        if "last_tool" in metadata:
            chain.append(metadata["last_tool"])
            if len(chain) > 5:
                chain = chain[-5:] # Keep last 5

        session.working_memory = wm
        session.analysis_chain = chain
        session.save(update_fields=['working_memory', 'analysis_chain'])

    @staticmethod
    def extract_metadata_footer(text: str, session: AnalysisSession = None) -> tuple:
        """
        Parses multi-line [METADATA] footer.
        Example:
        [METADATA]
        type: correlation
        target: Age
        columns: Age|Income
        
        Returns (clean_text, metadata_dict)
        """
        metadata = {}
        pattern = r"\[METADATA\]\s*([\s\S]*)$"
        match = re.search(pattern, text)
        
        clean_text = text
        if match:
            footer_block = match.group(1).strip()
            # Extract lines
            lines = footer_block.split('\n')
            for line in lines:
                if ':' in line:
                    key, val = line.split(':', 1)
                    key = key.strip().lower()
                    val = val.strip()
                    
                    # 3. Validation Guard: Skip empty values to prevent corruption
                    if not val:
                        continue

                    if key == "columns":
                        metadata["active_columns"] = [c.strip() for c in val.split('|')][:MemoryManager.ACTIVE_COLUMNS_MAX]
                    elif key == "type":
                        metadata["last_result_type"] = val
                    elif key == "target":
                        metadata["last_target_column"] = val

            # Authority Layer Update (v3.2)
            if metadata and session:
                session.last_valid_metadata = metadata
                session.save(update_fields=['last_valid_metadata'])
                logger.info("Updated last_valid_metadata sigil.")
            else:
                logger.warning("Metadata extraction returned empty. Fallback to last_valid_metadata will be used in resolver.")

            # Remove the metadata block from the user-facing text
            clean_text = text[:match.start()].strip()
            
        return clean_text, metadata

    @staticmethod
    def reset_memory(session: AnalysisSession):
        """
        Hard wipe of all memory layers for dataset switches or explicit resets.
        """
        logger.info(f"Performing total memory wipe for session {session.id}")
        session.working_memory = {}
        session.memory_summary = ""
        session.analysis_chain = []
        session.last_valid_metadata = {}
        session.last_summary_at_message = 0
        session.save()
