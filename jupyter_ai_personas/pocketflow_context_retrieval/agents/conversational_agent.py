"""
agents/conversational_agent.py - PocketFlow Conversational Agent with Bedrock LLM Integration

Implements the PocketFlow agent pattern with proper decision nodes, action spaces, 
and LLM integration using Jupyter AI's model manager configuration.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml

from pocketflow import Node, Flow

logger = logging.getLogger(__name__)

class ConversationalDecisionNode(Node):
    """
    PocketFlow decision node that analyzes user messages and decides actions.
    Implements the agent pattern from PocketFlow documentation.
    """
    
    def __init__(self, llm_provider=None, **kwargs):
        super().__init__(**kwargs)
        self.llm_provider = llm_provider
        
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for conversational decision making."""
        message = shared.get("user_message", "")
        conversation_history = shared.get("conversation_history", [])
        
        # Build minimal, relevant context (per PocketFlow best practices)
        recent_context = conversation_history[-3:] if conversation_history else []
        
        return {
            "current_message": message,
            "recent_context": recent_context,
            "available_actions": self._get_action_space(),
            "timestamp": datetime.now().isoformat()
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conversational decision using LLM."""
        try:
            # Build decision prompt using PocketFlow agent pattern
            decision_prompt = self._build_decision_prompt(prep_res)
            
            # Call LLM for structured decision
            if self.llm_provider:
                decision_response = self._call_llm_for_decision(decision_prompt)
                parsed_decision = self._parse_decision_response(decision_response)
            else:
                # Fallback to rule-based decision
                parsed_decision = self._rule_based_decision(prep_res["current_message"])
            
            return {
                "decision_successful": True,
                "chosen_action": parsed_decision.get("action", "conversational_response"),
                "action_parameters": parsed_decision.get("parameters", {}),
                "reasoning": parsed_decision.get("reasoning", "Rule-based decision"),
                "confidence": parsed_decision.get("confidence", 0.8)
            }
            
        except Exception as e:
            logger.error(f"❌ Decision node failed: {e}")
            return {
                "decision_successful": False,
                "chosen_action": "error_response",
                "action_parameters": {"error": str(e)},
                "reasoning": "Fallback due to error"
            }
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Route to next node based on decision."""
        action = exec_res.get("chosen_action", "error_response")
        
        # Store decision context in shared data
        shared["agent_decision"] = exec_res
        shared["next_action"] = action
        
        # Return next node route
        if action == "conversational_response":
            return "conversation"
        elif action == "analysis_request":
            return "analysis"
        elif action == "mixed_interaction":
            return "mixed"
        else:
            return "error"
    
    def _get_action_space(self) -> List[Dict[str, Any]]:
        """Define available actions for the agent (PocketFlow pattern)."""
        return [
            {
                "name": "conversational_response",
                "description": "Handle friendly conversation, greetings, questions about capabilities",
                "parameters": ["response_type", "personality_mode"],
                "examples": ["hello", "how are you", "what can you do"]
            },
            {
                "name": "analysis_request", 
                "description": "Process request for notebook analysis or technical help",
                "parameters": ["analysis_type", "focus_areas", "urgency"],
                "examples": ["analyze my code", "help optimize pandas", "find examples"]
            },
            {
                "name": "mixed_interaction",
                "description": "Handle messages with both conversational and analytical elements", 
                "parameters": ["conversational_part", "analytical_part"],
                "examples": ["hi, can you help me optimize this code?"]
            },
            {
                "name": "enhancement_request",
                "description": "Improve or personalize existing analysis results",
                "parameters": ["enhancement_type", "focus_areas"],
                "examples": ["make this more focused on performance", "explain this better"]
            }
        ]
    
    def _build_decision_prompt(self, prep_res: Dict[str, Any]) -> str:
        """Build structured prompt for LLM decision making."""
        message = prep_res["current_message"]
        actions = prep_res["available_actions"]
        context = prep_res.get("recent_context", [])
        
        # Convert actions to YAML format (PocketFlow structured output pattern)
        actions_yaml = yaml.dump(actions, default_flow_style=False)
        
        context_str = ""
        if context:
            context_str = f"""
RECENT CONVERSATION CONTEXT:
{yaml.dump(context, default_flow_style=False)}
"""
        
        prompt = f"""You are an intelligent PocketFlow conversational agent. Your job is to analyze the user's message and decide the best way to respond.

USER MESSAGE: "{message}"
{context_str}
AVAILABLE ACTIONS:
{actions_yaml}

INSTRUCTIONS:
- Analyze the user's intent naturally - don't rely on keyword matching
- Consider the conversation context and flow
- Choose the action that will provide the most helpful response
- Be intelligent about mixed requests (e.g., "Hi, can you help me optimize my code?")

Examples:
- "Hello!" → conversational_response (greeting)
- "Can you analyze my pandas code?" → analysis_request (needs technical analysis) 
- "Hi, I need help with my notebook performance" → mixed_interaction (greeting + technical)
- "Thanks! Now make this more focused on performance" → enhancement_request (improving previous response)

Respond in YAML format:
```yaml
action: <action_name>
parameters:
  response_type: <type>
  focus_area: <area_if_applicable>
  personality_mode: <friendly|professional|helpful>
reasoning: <your_reasoning>
confidence: <0.0_to_1.0>
```"""
        
        return prompt
    
    def _call_llm_for_decision(self, prompt: str) -> str:
        """Call LLM using Jupyter AI's model provider."""
        try:
            response = self.llm_provider.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            raise
    
    def _parse_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse structured YAML response from LLM."""
        try:
            # Extract YAML from markdown code blocks if present
            if "```yaml" in response:
                yaml_start = response.find("```yaml") + 7
                yaml_end = response.find("```", yaml_start)
                yaml_content = response[yaml_start:yaml_end].strip()
            else:
                yaml_content = response
            
            # Parse YAML
            parsed = yaml.safe_load(yaml_content)
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Failed to parse LLM response: {e}")
            # Fallback to rule-based
            return self._rule_based_decision(response)
    
    def _rule_based_decision(self, message: str) -> Dict[str, Any]:
        """Fallback rule-based decision making."""
        message_lower = message.lower().strip()
        
        # Simple pattern matching
        if any(word in message_lower for word in ["hello", "hi", "hey", "thanks", "who are you"]):
            return {
                "action": "conversational_response",
                "parameters": {"response_type": "greeting", "personality_mode": "friendly"},
                "reasoning": "Detected conversational greeting",
                "confidence": 0.9
            }
        elif any(word in message_lower for word in ["analyze", "help", "optimize", "code", "notebook"]):
            return {
                "action": "analysis_request", 
                "parameters": {"analysis_type": "general", "urgency": "medium"},
                "reasoning": "Detected analysis request",
                "confidence": 0.8
            }
        else:
            return {
                "action": "conversational_response",
                "parameters": {"response_type": "general", "personality_mode": "helpful"},
                "reasoning": "Default conversational response",
                "confidence": 0.7
            }


class ConversationResponseNode(Node):
    """Handle conversational responses with personality."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def exec(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversational response."""
        decision = shared.get("agent_decision", {})
        action_params = decision.get("action_parameters", {})
        message = shared.get("user_message", "")
        
        response_type = action_params.get("response_type", "general")
        personality_mode = action_params.get("personality_mode", "friendly")
        
        # Generate response based on type
        if response_type == "greeting":
            response = self._generate_greeting_response(message, personality_mode)
        elif response_type == "capabilities":
            response = self._generate_capabilities_response()
        elif response_type == "general":
            response = self._generate_general_response(message, personality_mode)
        else:
            response = self._generate_default_response(message)
        
        return {
            "response_generated": True,
            "response_content": response,
            "response_type": response_type,
            "personality_used": personality_mode
        }
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]):
        """Store final response."""
        shared["final_response"] = exec_res["response_content"]
        shared["response_ready"] = True
        return "default"
    
    def _generate_greeting_response(self, message: str, personality: str) -> str:
        """Generate personalized greeting."""
        return f"""# 👋 Hello! Great to see you!

I'm your **PocketFlow Context Assistant** - ready to help with intelligent data science analysis!

## 🚀 **What I can do for you:**
- 🔍 **Deep notebook analysis** with workflow detection
- 📚 **Smart research** through the Python Data Science Handbook
- 💡 **Personalized recommendations** tailored to your specific needs
- 💬 **Friendly conversation** about your data science challenges

**What would you like to explore today?** ✨"""
    
    def _generate_capabilities_response(self) -> str:
        """Generate capabilities overview."""
        return """# 🧠 My PocketFlow-Powered Capabilities

## 🔍 **Advanced Analysis:**
- Deep notebook understanding with workflow stage detection
- Code complexity assessment and optimization suggestions
- Library usage pattern analysis

## 📚 **Intelligent Research:**
- Multi-query search through Python Data Science Handbook
- Quality filtering with advanced relevance scoring
- Context-aware content matching

## 💬 **Smart Interaction:**
- Natural conversation with technical expertise
- Adaptive responses based on your needs
- Context memory for better continuity

**Ready to put my intelligence to work!** 🚀"""
    
    def _generate_general_response(self, message: str, personality: str) -> str:
        """Generate general conversational response."""
        return f"""# 💬 Thanks for reaching out!

You said: *"{message}"*

I'm here to help with both friendly conversation and serious data science analysis!

**What would you like to do:**
- 💬 Keep chatting - ask me anything!
- 🔍 Analyze a notebook or workflow  
- 📚 Search for specific techniques
- ❓ Learn about my capabilities

**Just let me know what's on your mind!** 🚀"""
    
    def _generate_default_response(self, message: str) -> str:
        """Default fallback response."""
        return f"""# 🤖 I'm here to help!

**Let me know what you'd like to do:**
- Chat about your data science work
- Analyze notebooks and code
- Find relevant examples and techniques
- Get personalized recommendations

**What interests you most?** ✨"""


class IntelligentConversationalAgent:
    """
    PocketFlow-based conversational agent with LLM integration.
    
    Implements the agent design pattern with decision nodes, action spaces,
    and proper flow management using Jupyter AI's Bedrock model manager.
    """
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.conversation_flow = self._build_conversation_flow()
        self.conversation_history = []
        
    def _build_conversation_flow(self) -> Flow:
        """Build PocketFlow conversational agent flow."""
        # Create nodes
        decision_node = ConversationalDecisionNode(llm_provider=self.llm_provider)
        conversation_node = ConversationResponseNode()
        
        # Set up flow routing
        decision_node.set_next("conversation", conversation_node)
        decision_node.set_next("error", conversation_node)  # Error handling
        
        # Create flow
        flow = Flow(start=decision_node)
        
        return flow
    
    async def handle_message(self, message: str, raw_analysis: Dict = None, context_info: Dict = None) -> str:
        """
        Handle message using PocketFlow agent pattern.
        
        Args:
            message: User's message
            raw_analysis: Optional raw analysis results to enhance
            
        Returns:
            Agent response
        """
        try:
            # Prepare shared data for flow
            shared_data = {
                "user_message": message,
                "conversation_history": self.conversation_history,
                "raw_analysis": raw_analysis,
                "context_info": context_info or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Run PocketFlow agent
            self.conversation_flow.run(shared_data)
            
            # Get response
            response = shared_data.get("final_response", "I'm here to help! What would you like to do?")
            
            # Update conversation history
            self._update_conversation_history(message, response)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Conversational agent failed: {e}")
            return self._create_error_response(str(e))
    
    def _update_conversation_history(self, user_message: str, agent_response: str):
        """Update conversation history with context window management."""
        self.conversation_history.append({
            "user": user_message,
            "agent": agent_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep last 10 interactions (PocketFlow minimal context principle)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _create_error_response(self, error_msg: str) -> str:
        """Create friendly error response."""
        return f"""# 😅 **Something went a bit sideways!**

**What happened:** {error_msg}

## 🛠️ **Let's get back on track:**

1. **Try rephrasing** - Sometimes I understand better with different wording
2. **Be more specific** - More context helps me help you better  
3. **Start simple** - We can always dive deeper step by step

**I'm still here and ready to help!** What would you like to try? 🚀"""