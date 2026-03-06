import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_ollama import ChatOllama
from Agents.menuAgent import MenuAgent
from Agents.orderAgent import OrderAgent
from Agents.routerAgent import RouterAgent, RouteDecision
from Agents.upsellingAgent import UpsellingAgent
from SharedMemory import SharedMemory
from Tools.validator import sanitize_input
from config import Config
import uuid
from typing import Tuple, Dict, Any

class NewCoordinatorAgent:
    """
    New Coordinator Agent that implements Router-first architecture
    All user inputs go through Router Agent first for intelligent routing
    """
    def __init__(self):
        self.llm = ChatOllama(
            model = Config.MODEL_NAME,
            temperature = Config.MODEL_TEMPERATURE
        )

        self.shared_memory = SharedMemory()
        self.router_agent = RouterAgent(llm = self.llm)
        self.menu_agent = MenuAgent(llm = self.llm)
        self.order_agent = OrderAgent(llm = self.llm, shared_memory = self.shared_memory)
        self.upselling_agent = UpsellingAgent(llm = self.llm)

        self.session_id = str(uuid.uuid4())

    def process_user_input(self,user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Main conversation processing method -  all inputs go through Router first"""
        try:
            if self._is_cancel_intent(user_input, None):
                response = self._handle_cancel_request(user_input, None)
                self.shared_memory.add_to_history(user_input, response, "coordinator")
                return response, self.shared_memory.to_dict()
            
            conversation_context = self.shared_memory.get_context_summary()
            route_decision = self.router_agent.route_conversation(user_input, conversation_context)

            if self._is_cancel_intent(user_input, route_decision):
                response = self._handle_cancel_request(user_input, route_decision)
                self.shared_memory.add_to_history(user_input, response, "coordinator")
                return response, self.shared_memory.to_dict()

            if route_decision.agent == "human" or self.shared_memory.needs_human_intervention:
                return self._handle_human_intervention(user_input, route_decision)

            response = self._execute_agent_action(user_input, route_decision)
            self.shared_memory.add_to_history(user_input, response, route_decision.agent)
            response = self._post_process_response(response, route_decision)
            return response, self.shared_memory.to_dict()
        
        except Exception as e:
            error_message = f"I encountered an error: {str(e)}. Let me try to help you differently."
            self.shared_memory.increment_error(str(e))
            return error_message, self.shared_memory.to_dict()

    def _execute_agent_action(self, user_input: str, route_decision: RouteDecision) -> str:
        """Execute the action based on Router's decision"""
        if route_decision.needs_clarification:
            self.shared_memory.pending_clarifications.append(route_decision.clarification_question)
            return route_decision.clarification_question or "Could you please clarify what you're looking for?"

        if route_decision.agent =="menu":
            return self._handle_menu_request(user_input, route_decision)
        elif route_decision.agent == "order":
            return self._handle_order_request(user_input, route_decision)
            
        elif route_decision.agent == "upselling":
            return self._handle_upselling_request(user_input, route_decision)
            
        elif route_decision.agent == "finalization":
            return self._handle_finalization_request(user_input, route_decision)
            
        elif route_decision.agent == "delivery":
            return self._handle_delivery_request(user_input, route_decision)
            
        else:
            # Default to menu for unclear requests
            return self._handle_menu_request(user_input, route_decision)

    def _handle_menu_request(self, user_input: str, route_decision: RouteDecision) -> str:
        """Handle menu-related requests"""
        self.shared_memory.set_customer_intent("BROWSING", "User browsing menu")
        
        if "menu" in user_input.lower() or "show" in user_input.lower():
            self.shared_memory.menu_displayed = True
            return self.menu_agent.display_menu()
        elif "recommend" in user_input.lower():
            return self.menu_agent.get_recommendations()
        else:
            return self.menu_agent.handle_menu_query(user_input)

    def _handle_order_request(self, user_input: str, route_decision: RouteDecision)-> str:
        """Handle order-related requests using extracted items form Router"""
        self.shared_memory.set_customer_intent("ORDERING","User placing or modifying order")

        # if router classified it as modification, apply modifications deterministically
        if route_decision.user_intent == "MODIFY_ORDER":
            modification_result = self.order_agent.handle_order_modification(user_input)

            if self.shared_memory.conversation_stage == "awaiting_delivery":
                return f"{modification_result}\n\nShall we proceed with delivery or pickup?"
            return modification_result
        
        if route_decision.extracted_items:
            result = self.order_agent.process_order_with_extracted_items(
                user_input, route_decision.extracted_items
            )
            if result.success:
                response_parts = [result.message]

                if result.added_items:
                    order_summary = self.order_agent.get_order_summary()
                    response_parts.append(f"\nCurrent order total: ${order_summary['totals']['total']:.2f}")
                
                if result.failed_items:
                    failed_names = [item.get("requested_name", "") for item in result.failed_items]
                    response_parts.append(f"\nCouldn't find: {', '.join(failed_names)}")
                    response_parts.append("Would you like to see our menu for available items?")
                
                return "\n".join(response_parts)
            else:
                return result.message
        else:
            return "I'd be happy to take your order! What would you like to order from our menu?"

    def _handle_upselling_request(self,user_input: str, route_decisions: RouteDecision) -> str:
        """Handle upselling based on current order"""
        if not self.shared_memory.current_order:
            return "Would you like to add something to your order?"
        if self.shared_memory.upsell_attempts >= self.shared_memory.max_upsell_attempts:
            return "Would you like anything else, or shall we proceed with your order?"

        order_items = []
        for item in self.shared_memory.current_order:
            mock_item = type('MockItem',(),{
                'name': item.get('name',''),
                'price': item.get('price',0),
                'quantity':item.get('quantity',1)
            })
            order_items.append(mock_item)
        mock_order = type('MockOrder',(),{'items':order_items})()
        try:
            upsell_response = self.upselling_agent.suggest_upsell(mock_order)
            self.shared_memory.upsell_attempts += 1
            return upsell_response
        except:
            return "Would you like to add any drinks or sides to complete your order?"

    def _handle_finalization_request(self, user_input: str, route_decision: RouteDecision) -> str:
        """Handle order finalization and completion"""

        if route_decision.wants_order_change and route_decision.user_intent == "CANCEL_ORDER":
            self.shared_memory.clear_order()
            self.shared_memory.set_customer_intent("GREETING", "Order cancelled by user")
            self.shared_memory.conversation_stage = "greeting"
            return "Your order has been cancelled. Would you like to start a new order?"
        
        self.shared_memory.set_customer_intent("FINALIZING", "User finalizing order")
        validation = self.order_agent.validate_order_completion()
        
        if not validation["ready"]:
            return validation["message"]
        
        order_summary = self.order_agent.get_order_summary()
        
        response_parts = [
            "**FINAL ORDER CONFIRMATION**",
            "",
            "Here's your complete order:",
            ""
        ]
        
        for item in order_summary['items']:
            if item['quantity'] == 1:
                response_parts.append(f"• {item['name']} - ${item['unit_price']:.2f}")
            else:
                response_parts.append(f"• {item['quantity']}x {item['name']} - ${item['unit_price']:.2f} each")
            
            if item['customizations']:
                response_parts.append(f"  └ Customizations: {', '.join(item['customizations'])}")
        
        response_parts.extend([
            "",
            f"**Total: ${order_summary['totals']['total']:.2f}** (includes tax)",
            "",
            "Would you like this delivered or would you prefer pickup?"
        ])
        
        self.shared_memory.conversation_stage = "awaiting_delivery"
        self.shared_memory.set_customer_intent("DELIVERY_METHOD", "Waiting for delivery method choice")
        
        return "\n".join(response_parts)

    def _handle_delivery_request(self, user_input: str, route_decision: RouteDecision) -> str:
        """Handle delivery method selection and processing"""
        
        if route_decision.wants_order_change:
            if route_decision.user_intent == "MODIFY_ORDER":
                modification_result = self.order_agent.handle_order_modification(user_input)
                self.shared_memory.conversation_stage = "awaiting_delivery"
                self.shared_memory.set_customer_intent("DELIVERY_METHOD", "Waiting for delivery method choice")
                return f"{modification_result}\n\nShall we proceed with delivery or pickup?"
                
            elif route_decision.user_intent == "CANCEL_ORDER":
                self.shared_memory.clear_order()
                self.shared_memory.set_customer_intent("GREETING", "Order cancelled by user")
                self.shared_memory.conversation_stage = "greeting"
                return "Your order has been cancelled. Would you like to start a new order?"

        if route_decision.delivery_method:
            self.shared_memory.delivery_method = route_decision.delivery_method
            self.shared_memory.conversation_stage = "completed"
            self.shared_memory.set_customer_intent("COMPLETED", f"Order completed with {route_decision.delivery_method}")
            
            if route_decision.delivery_method == "delivery":
                response_parts = [
                    "Perfect! Your order will be delivered.",
                    "",
                    "**ORDER SUMMARY**",
                    f"• Order Total: ${self.shared_memory.order_total:.2f}",
                    f"• Delivery Method: Delivery",
                    f"• Estimated Delivery Time: 30-45 minutes",
                    "",
                    "Thank you for your order! We'll start preparing it right away.",
                    "You'll receive updates on your order status."
                ]
            else:  # pickup
                response_parts = [
                    "Great! Your order will be ready for pickup.",
                    "",
                    "**ORDER SUMMARY**",
                    f"• Order Total: ${self.shared_memory.order_total:.2f}",
                    f"• Pickup Method: Pickup",
                    f"• Estimated Pickup Time: 15-20 minutes",
                    "",
                    "Thank you for your order! We'll start preparing it right away.",
                    "We'll notify you when it's ready for pickup."
                ]
            
            return "\n".join(response_parts)
        
        return (
            "I didn't quite catch that. Please let me know:\n"
            "• Type 'delivery' if you'd like it delivered\n"
            "• Type 'pickup' if you'd prefer to pick it up\n"
            "• Or feel free to add more items to your order first!"
        )

    def _handle_human_intervention(self, user_input: str, route_decision: RouteDecision) -> str:
        """Handle cases requiring human intervention"""
        intervention_messages = [
            "I understand this might be a complex request. Let me connect you with our human assistant.",
            f"Request: {user_input}",
            f"Reason: {self.shared_memory.intervention_reason or 'Complex query requiring human assistance'}",
            "",
            "In the meantime, I can still help with basic menu questions or simple orders."
        ]
        
        return "\n".join(intervention_messages)

    def _post_process_response(self, response: str, route_decision: RouteDecision) -> str:
        """Post-process response and determine next steps"""
        
        if (route_decision.agent == "order" and 
            self.shared_memory.current_order and 
            self.shared_memory.upsell_attempts < self.shared_memory.max_upsell_attempts):
            upsell_suggestions = [
                "\n\nWould you like to add any drinks or sides?",
                "\n\nHow about some appetizers to start?",
                "\n\nAny beverages to go with that?"
            ]
            
            import random
            response += random.choice(upsell_suggestions)
            self.shared_memory.upsell_attempts += 1
        
        return response

    def _is_cancel_intent(self, user_input: str, route_decision: RouteDecision) -> bool:
        """Detect full-order cancellation regardless of conversation stage."""
        try:
            if getattr(route_decision, "user_intent", None) == "CANCEL_ORDER":
                return True
        except Exception:
            pass

        text = sanitize_input(user_input).lower().strip()
        
        simple_cancel_words = ["cancel", "stop", "quit", "exit", "nevermind", "never mind", "forget it"]
        if text in simple_cancel_words:
            return True
        
        cancel_phrases = [
            "cancel order",
            "cancel my order", 
            "cancel the order",
            "i don't want to order anymore",
            "i don't want to order any more",
            "i don't want this order",
            "nevermind the order",
            "never mind the order",
            "forget the order",
            "void the order",
            "stop the order",
            "end the order",
            "no thanks",
            "not interested"
        ]
        return any(p in text for p in cancel_phrases)

    def _handle_cancel_request(self, user_input: str, route_decision: RouteDecision) -> str:
        """Clear the current order and reset state."""
        if not self.shared_memory.current_order:
            self.shared_memory.set_customer_intent("GREETING", "User requested cancel with no active order")
            self.shared_memory.conversation_stage = "greeting"
            return "There is no active order to cancel. Would you like to start a new order or see the menu?"
        
        self.shared_memory.clear_order()
        self.shared_memory.set_customer_intent("GREETING", "Order cancelled by user")
        self.shared_memory.conversation_stage = "greeting"
        return "Your order has been cancelled. Would you like to start a new order or see the menu?"


    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state for monitoring"""
        return {
            "session_id": self.session_id,
            "customer_intent": self.shared_memory.customer_intent,
            "conversation_stage": self.shared_memory.conversation_stage,
            "order_items": len(self.shared_memory.current_order),
            "order_total": self.shared_memory.order_total,
            "last_agent": self.shared_memory.last_agent,
            "needs_intervention": self.shared_memory.needs_human_intervention,
            "upsell_attempts": self.shared_memory.upsell_attempts,
            "error_count": self.shared_memory.error_count
        }

    def handle_intelligent_suggestions(self, partial_input: str) -> str:
        """Get intelligent suggestions for partial input"""
        suggestions = self.router_agent.get_intelligent_suggestions(partial_input)
        
        if suggestions:
            response_parts = [
                "I think you might be looking for:",
                ""
            ]
            
            for i, suggestion in enumerate(suggestions, 1):
                response_parts.append(f"{i}. {suggestion}")
            
            response_parts.append("\nWhich one interests you?")
            return "\n".join(response_parts)
        else:
            return "I'm not sure what you're looking for. Would you like to see our menu?"