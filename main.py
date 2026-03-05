import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from Agents.coordinatorAgent import NewCoordinatorAgent
import uuid

class RestaurantAIAgent:
    def __init__(self):
        """Initialize the Router-based Restaurant AI Agent system"""
        
        self.coordinator = NewCoordinatorAgent()
        self.session_id = str(uuid.uuid4())
        print("\n" + "="*60)
        print("Restaurant AI Agent")
        print("="*60)
        print("\nCommands: /help, /menu, /reset, /debug, /state, quit")

    def start_conversation(self):
        """Start an interactive conversation with the customer"""
        
        response, conversation_state = self.coordinator.process_user_input("hello")
        print(f"\nAI: {response}")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\nAI: Thanks for visiting AI Bistro. Have a great day!")
                    break
                
                if not user_input:
                    print("I didn't catch that. Please say something.")
                    continue
                
                if user_input.startswith("/"):
                    handled = self._handle_command(user_input)
                    if handled:
                        continue
                
                response, conversation_state = self.coordinator.process_user_input(user_input)
                
                print(f"\nAI: {response}")
                
                if self.debug_mode:
                    self._show_debug_info(conversation_state)
                
                if conversation_state.get("customer_intent") == "COMPLETED":
                    details = self.get_order_details()
                    items = details.get("items", [])
                    totals = details.get("totals", {})
                    if items:
                        self._print_order_summary(items, totals)
                    
                    new_order = input("\nStart a new order? (y/n): ").strip().lower()
                    if new_order in ['yes', 'y']:
                        self.coordinator.reset_conversation()
                        print("ℹStarting a new order...")
                        response, _ = self.coordinator.process_user_input("hello")
                        print(f"\nAI: {response}")
                    else:
                        print("\nAI: Thank you for choosing AI Bistro!")
                        break
                
                if conversation_state.get("needs_intervention"):
                    print("A human operator will assist shortly.")
                
            except KeyboardInterrupt:
                print("\nAI: Thanks for visiting us.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("ℹPlease try again or contact support.")

    def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        name = cmd.lower().strip()
        
        if name in ("/help", "/h"):
            print("\n" + "="*60)
            print("AVAILABLE COMMANDS")
            print("="*60)
            print("  /menu   - Show the menu")
            print("  /state  - Show current conversation state")
            print("  /reset  - Reset the conversation")
            print("  /debug  - Toggle debug info display")
            print("  quit    - Exit the assistant")
            print("="*60)
            return True
        
        if name == "/menu":
            print("\n" + "="*60)
            print("MENU")
            print("="*60)
            menu_display = self.coordinator.menu_agent.display_menu()
            print(menu_display)
            return True
        
        if name == "/reset":
            self.reset_conversation()
            print("Conversation reset.")
            response, _ = self.coordinator.process_user_input("hello")
            print(f"\nAI: {response}")
            return True
        
        if name == "/debug":
            self.debug_mode = not self.debug_mode
            status = "ON" if self.debug_mode else "OFF"
            print(f"Debug mode: {status}")
            return True
        
        if name == "/state":
            state = self.coordinator.shared_memory.to_dict()
            self._show_debug_info(state)
            return True
        
        print("Unknown command. Type /help for options.")
        return True

    def _show_debug_info(self, conversation_state):
        """Show debug information for development"""
        print("\n" + "─"*60)
        print("DEBUG INFO")
        print("─"*60)
        print(f"Intent:       {conversation_state.get('customer_intent', 'Unknown')}")
        print(f"Stage:        {conversation_state.get('conversation_stage', 'Unknown')}")
        print(f"Order Items:  {len(conversation_state.get('current_order', []))}")
        print(f"Total:        ${conversation_state.get('order_total', 0):.2f}")
        print(f"Last Agent:   {conversation_state.get('last_agent', 'None')}")
        print("─"*60)

    def _print_order_summary(self, items, totals):
        """Print order summary"""
        print("\n" + "="*60)
        print("ORDER SUMMARY")
        print("="*60)
        
        for item in items:
            qty = item.get('quantity', 1)
            name = item.get('name', 'Unknown')
            price = item.get('price', 0)
            total_price = price * qty
            
            print(f"  {qty}x {name:<40} ${total_price:>8.2f}")
            
            customizations = item.get('customizations', [])
            if customizations:
                print(f"      └─ {', '.join(customizations)}")
        
        print("─"*60)
        print(f"Subtotal:  ${totals.get('subtotal', 0):.2f}")
        print(f"Tax:       ${totals.get('tax', 0):.2f}")
        print(f"Total:     ${totals.get('total', 0):.2f}")
        print("="*60)

    def process_single_request(self, user_input: str) -> str:
        """Process a single request (useful for API integration)"""
        try:
            response, _ = self.coordinator.process_user_input(user_input)
            return response
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}. Please try again."

    def get_order_details(self) -> dict:
        """Get current order details for external systems"""
        conversation_state = self.coordinator.get_conversation_state()
        
        return {
            "session_id": self.session_id,
            "order_id": conversation_state.get("session_id"),
            "items": self.coordinator.shared_memory.current_order,
            "totals": {
                "subtotal": sum(item.get("price", 0) * item.get("quantity", 1) 
                              for item in self.coordinator.shared_memory.current_order),
                "tax": sum(item.get("price", 0) * item.get("quantity", 1) 
                          for item in self.coordinator.shared_memory.current_order) * 0.08,
                "total": self.coordinator.shared_memory.order_total
            },
            "status": conversation_state.get("customer_intent", "UNKNOWN"),
            "needs_intervention": conversation_state.get("needs_intervention", False),
            "routing_info": {
                "last_agent": conversation_state.get("last_agent"),
                "upsell_attempts": conversation_state.get("upsell_attempts", 0),
                "error_count": conversation_state.get("error_count", 0)
            }
        }

    def get_intelligent_suggestions(self, partial_input: str) -> str:
        """Get intelligent suggestions for partial/unclear input"""
        return self.coordinator.handle_intelligent_suggestions(partial_input)

    def simulate_human_intervention(self, reason: str = "Testing intervention"):
        """Simulate human intervention for testing purposes"""
        self.coordinator.shared_memory.trigger_human_intervention(reason)
        print(f"Human intervention triggered: {reason}")

    def reset_conversation(self):
        """Reset the conversation state"""
        self.coordinator.reset_conversation()
        self.session_id = str(uuid.uuid4())

    def get_conversation_analytics(self) -> dict:
        state = self.coordinator.get_conversation_state()
        memory = self.coordinator.shared_memory
        
        return {
            "session_info": {
                "session_id": self.session_id,
                "duration_seconds": (memory.last_activity - memory.session_start).total_seconds(),
                "total_interactions": len(memory.conversation_history)
            },
            "order_analytics": {
                "items_count": len(memory.current_order),
                "order_value": memory.order_total,
                "avg_item_price": memory.order_total / max(len(memory.current_order), 1)
            },
            "agent_usage": {
                "last_agent": state.get("last_agent"),
                "upsell_attempts": state.get("upsell_attempts", 0),
                "human_interventions": 1 if state.get("needs_intervention") else 0
            },
            "conversation_flow": {
                "current_intent": state.get("customer_intent"),
                "current_stage": state.get("conversation_stage"),
                "errors_encountered": state.get("error_count", 0)
            }
        }

def main():
    agent = RestaurantAIAgent()
    agent.start_conversation()
        
if __name__ == "__main__":
    main()