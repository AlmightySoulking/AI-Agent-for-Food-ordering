from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
import operator
from Agents.coordinatorAgent import NewCoordinatorAgent
from IPython.display import Image, display
import argparse

class RestaurantState(TypedDict):
    messages: Annotated[list, add_messages]
    current_order: dict
    conversation_stage: str
    customer_info: dict
    order_total: float
    upsell_attempts: int
    menu_displayed: bool
    customer_intent: str
    last_agent: str
    needs_intervention: bool
    router_decision: dict

class RestaurantGraph:
    def __init__(self, coordinator_agent):
        """Initialize with the new CoordinatorAgent"""
        self.coordinator = coordinator_agent
        self.graph = self.build_graph()

    def build_graph(self):
        """Build the conversation flow graph showing Router Agent as central hub"""
        workflow = StateGraph(RestaurantState)

        workflow.add_node("greeting", self._greeting_node)
        workflow.add_node("router_agent", self._router_agent_node)
        workflow.add_node("menu_agent", self._menu_agent_node)
        workflow.add_node("order_agent", self._order_agent_node)
        workflow.add_node("upselling_agent", self._upselling_agent_node)
        workflow.add_node("finalization_agent", self._finalization_agent_node)
        workflow.add_node("delivery_agent", self._delivery_agent_node)
        workflow.add_node("human_intervention", self._human_intervention_node)
        workflow.add_node("completion", self._completion_node)
        
        workflow.add_edge(START, "greeting")
        workflow.add_edge("greeting", "router_agent")
        
        workflow.add_conditional_edges(
            "router_agent",
            self._route_from_router,
            {
                "menu_agent": "menu_agent",
                "order_agent": "order_agent",
                "upselling_agent": "upselling_agent",
                "finalization_agent": "finalization_agent",
                "delivery_agent": "delivery_agent",
                "human_intervention": "human_intervention"
            }
        )

        workflow.add_edge("menu_agent", "router_agent")
        workflow.add_edge("order_agent", "router_agent")
        workflow.add_edge("upselling_agent", "router_agent")
        workflow.add_edge("finalization_agent", "router_agent")
        workflow.add_edge("delivery_agent", "router_agent")
        workflow.add_edge("human_intervention", "completion")
        workflow.add_edge("completion", END)

        return workflow.compile()
    
    def _greeting_node(self, state: RestaurantState) -> RestaurantState:
        """Handle initial greeting stage"""
        response, _ = self.coordinator.process_user_input("hello")
        state['messages'] = [HumanMessage(content=response)]
        state['current_order'] = {}
        state['conversation_stage'] = "start"
        state['order_total'] = 0.0
        state['upsell_attempts'] = 0
        state['menu_displayed'] = False
        print(f"AI: {response}")
        return state

    def _router_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Central Router Agent node - makes all routing decisions"""
        last_message = state["messages"][-1].content if state.get("messages") else "hello"

        conversation_context = self.coordinator.shared_memory.get_context_summary()
        route_decision = self.coordinator.router_agent.route_conversation(
            last_message, conversation_context
        )
        state["router_decision"] = route_decision
        state["customer_intent"] = route_decision.user_intent
        state["last_agent"] = "router"
        
        return state

    def _menu_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Menu Agent node - handles menu queries and recommendations"""
        last_message = state["messages"][-1].content
        response = self.coordinator._handle_menu_request(last_message, state['router_decision'])
        state['messages'].append(HumanMessage(content=response))
        state["last_agent"] = "menu"
        state["conversation_stage"] = "browsing"
        state["menu_displayed"] = True
        return state

    def _order_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Order Agent node - processes orders and item extraction"""
        last_message = state["messages"][-1].content
        response = self.coordinator._handle_order_request(last_message, state['router_decision'])
        state['messages'].append(HumanMessage(content=response))
        state["last_agent"] = "order"
        state["conversation_stage"] = "ordering"
        state['current_order'] = self.coordinator.shared_memory.current_order
        state['order_total'] = self.coordinator.shared_memory.order_total
        return state
    
    def _upselling_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Upselling Agent node - suggests complementary items"""
        last_message = state["messages"][-1].content
        response = self.coordinator._handle_upselling_request(last_message, state['router_decision'])
        state['messages'].append(HumanMessage(content=response))
        state["last_agent"] = "upselling"
        state["conversation_stage"] = "upselling"
        state["upsell_attempts"] = self.coordinator.shared_memory.upsell_attempts
        return state
    
    def _finalization_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Finalization Agent node - handles order completion"""
        last_message = state["messages"][-1].content
        response = self.coordinator._handle_finalization_request(last_message, state['router_decision'])
        state['messages'].append(HumanMessage(content=response))
        state["last_agent"] = "finalization"
        state["conversation_stage"] = "finalizing"
        return state
    
    def _delivery_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Delivery Agent node - handles delivery method selection"""
        last_message = state["messages"][-1].content
        response = self.coordinator._handle_delivery_request(last_message, state['router_decision'])
        state['messages'].append(HumanMessage(content=response))
        state["last_agent"] = "delivery"
        state["conversation_stage"] = "awaiting_delivery"
        return state
    
    def _human_intervention_node(self, state: RestaurantState) -> RestaurantState:
        """Handle human intervention cases"""
        last_message = state["messages"][-1].content
        response = self.coordinator._handle_human_intervention(last_message, state['router_decision'])
        state['messages'].append(HumanMessage(content=response))
        state["conversation_stage"] = "human_intervention"
        state["customer_intent"] = "HUMAN_NEEDED"
        state["last_agent"] = "human"
        state["needs_intervention"] = True
        return state

    def _completion_node(self, state: RestaurantState) -> RestaurantState:
        """Handle order completion"""
        print("Conversation completed.")
        return state
    
    def _route_from_router(self, state: RestaurantState) -> str:
        """Route from Router Agent based on its decision"""
        router_decision = state.get("router_decision")
        target_agent = router_decision.agent
        
        agent_mapping = {
            "menu": "menu_agent",
            "order": "order_agent",
            "upselling": "upselling_agent",
            "finalization": "finalization_agent",
            "delivery": "delivery_agent",
            "human": "human_intervention",
        }
        
        return agent_mapping.get(target_agent, "menu_agent")

def run_conversation(coordinator, restaurant_graph):
    """Run the conversation loop for the Restaurant AI Agent."""
    print("Restaurant AI Agent is ready. Type 'quit' to exit.")
    
    state = restaurant_graph.graph.invoke({"messages": []})
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        state['messages'].append(HumanMessage(content=user_input))
        state = restaurant_graph.graph.invoke(state)
        
        if state['messages'] and isinstance(state['messages'][-1], HumanMessage):
            print(f"AI: {state['messages'][-1].content}")
        
        coordinator.shared_memory.current_order = state.get('current_order', coordinator.shared_memory.current_order)
        coordinator.shared_memory.conversation_stage = state.get('conversation_stage', coordinator.shared_memory.conversation_stage)
        coordinator.shared_memory.order_total = state.get('order_total', coordinator.shared_memory.order_total)
        coordinator.shared_memory.upsell_attempts = state.get('upsell_attempts', coordinator.shared_memory.upsell_attempts)
        coordinator.shared_memory.menu_displayed = state.get('menu_displayed', coordinator.shared_memory.menu_displayed)
        coordinator.shared_memory.customer_intent = state.get('customer_intent', coordinator.shared_memory.customer_intent)
        coordinator.shared_memory.last_agent = state.get('last_agent', coordinator.shared_memory.last_agent)
        coordinator.shared_memory.needs_human_intervention = state.get('needs_intervention', coordinator.shared_memory.needs_human_intervention)

def generate_graph_image(restaurant_graph):
    """Generate a PNG image of the graph."""
    try:
        img_data = restaurant_graph.graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(img_data)
        print("Graph image saved to graph.png")
    except Exception as e:
        print(f"Error generating graph image: {e}")

def main():
    """Main function to run the Restaurant AI Agent graph or generate the graph image."""
    parser = argparse.ArgumentParser(description="Restaurant AI Agent")
    parser.add_argument("--generate-graph", action="store_true", help="Generate a PNG image of the graph")
    args = parser.parse_args()

    coordinator = NewCoordinatorAgent()
    restaurant_graph = RestaurantGraph(coordinator)

    if args.generate_graph:
        generate_graph_image(restaurant_graph)
    else:
        run_conversation(coordinator, restaurant_graph)

if __name__ == "__main__":
    graph = RestaurantGraph(NewCoordinatorAgent)
    png = graph.graph.get_graph().draw_mermaid_png()
    with open("workflow_graph.png", "wb") as f:
        f.write(png)


   # main()
