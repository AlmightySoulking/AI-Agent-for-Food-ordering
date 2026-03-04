from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

class RestaurantState(TypedDict):
    message: Annotated[list[BaseMessage],add_messages]
    current_order: dict
    conversation_stage: str
    customer_info: dict
    order_total: float
    upsell_attempts: float
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
        # Router node
        workflow.add_node("router_agent", self._router_agent_node)
        workflow.add_node("menu_agent",self._menu_agent_node)
        workflow.add_node("upselling_agent", self._upselling_agent_node)
        workflow.add_node("finalization_agent", self._finalization_agent_node)
        workflow.add_node("delivery_agent", self._delivery_agent_node)
        workflow.add_node("human_intervention", self._human_intervention_node)
        
        workflow.add_node("greeting", self._greeting_node)
        workflow.add_node("menu_browsing", self._menu_browsing_node)
        workflow.add_node("ordering", self._ordering_node)
        workflow.add_node("upselling", self._upselling_node)
        workflow.add_node("finalizing", self._finalizing_node)
        workflow.add_node("delivery_method", self._delivery_method_node)
        workflow.add_node("completion", self._completion_node)
        
        workflow.add_edge(START, "router_agent")

        workflow.add_conditional_edges(
            "router_agent",
            self._route_from_router,{
                "menu_agent":"menu_agent",
                "order_agent":"order_agent",
                "upselling_agent":"upselling_agent",
                "finalization_agent": "finalization_agent",
                "delivery_agent": "delivery_agent",
                "human_intervention": "human_intervention",
                "greeting": "greeting"
            }
        )

        workflow.add_edge("menu_agent","menu_browsing")
        workflow.add_edge("order_agent","ordering")
        workflow.add_edge("upselling_agent","upselling")
        workflow.add_edge("finalization_agent","finalizing")
        workflow.add_edge("delivery_agent","delivery_method")

        workflow.add_conditional_edges(
            "greeting",
            self._route_back_to_router,
            {
                "router_agent":"router_agent",
                "completion": "completion",
                "human_intervention": "human_intervention"
            }
        )

        workflow.add_conditional_edges(
            "menu_browsing",
            self._route_back_to_router,
            {
                "router_agent": "router_agent",
                "completion": "completion",
                "human_intervention": "human_intervention"
            }
        )
        
        workflow.add_conditional_edges(
            "ordering",
            self._route_back_to_router,
            {
                "router_agent": "router_agent",
                "completion": "completion",
                "human_intervention": "human_intervention"
            }
        )
        workflow.add_conditional_edges(
            "upselling",
            self._route_back_to_router,
            {
                "router_agent": "router_agent",
                "completion": "completion",
                "human_intervention": "human_intervention"
            }
        )
        
        workflow.add_conditional_edges(
            "finalizing",
            self._route_back_to_router,
            {
                "router_agent": "router_agent",
                "completion": "completion",
                "human_intervention": "human_intervention"
            }
        )
        
        workflow.add_conditional_edges(
            "delivery_method",
            self._route_back_to_router,
            {
                "router_agent": "router_agent",
                "completion": "completion",
                "human_intervention": "human_intervention"
            }
        )

        workflow.add_edge("completion",END)
        workflow.add_edge("human_intervention",END)

        return workflow.compile()
    
    def _router_agent_node(self,state: RestaurantState) -> RestaurantState:
        """Central Router Agent node - makes all routing decisions"""
        last_message = ""
        if state.get("messages"):
            last_message = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])

        if hasattr(self.coordinator,'router_agent'):
            conversation_context = {
                "current_order":state.get("current_order",{}),
                "conversation_stage": state.get("conversation_stage", "greeting"),
                "order_total": state.get("order_total", 0.0),
                "upsell_attempts": state.get("upsell_attempts", 0),
                "menu_displayed": state.get("menu_displayed", False)
           }
            route_decision = self.coordinator.router_agent.route_conversation(
                last_message,conversation_context
            )
            state["router_decision"] = {
                "target_agent": route_decision.agent,
                "confidence": route_decision.confidence,
                "user_intent": route_decision.user_intent,
                "extracted_items": route_decision.extracted_items,
                "needs_clarification": route_decision.needs_clarification
            }
            state["customer_intent"] = route_decision.user_intent
            state["last_agent"] = "router"
            if route_decision.extracted_items:
                state["current_order"] = {
                    "pending_items": route_decision.extracted_items,
                    "total": sum(item.get("price", 0) * item.get("quantity", 1) for item in route_decision.extracted_items)
                }
        
        return state

    def _menu_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Menu Agent node - handles menu queries and recommendations"""
        state["last_agent"] = "menu"
        state["conversation_stage"] = "browsing"
        state["menu_displayed"] = True
        return state
    def _order_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Order Agent node - processes orders and item extraction"""
        state["last_agent"] = "order"
        state["conversation_stage"] = "ordering"
        
        if state.get("router_decision", {}).get("extracted_items"):
            extracted_items = state["router_decision"]["extracted_items"]
            total = sum(item.get("price", 0) * item.get("quantity", 1) for item in extracted_items)
            state["order_total"] = state.get("order_total", 0) + total
            
            current_items = state.get("current_order", {}).get("items", [])
            current_items.extend(extracted_items)
            state["current_order"] = {
                "items": current_items,
                "total": state["order_total"]
            }
        
        return state
    
    def _upselling_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Upselling Agent node - suggests complementary items"""
        state["last_agent"] = "upselling"
        state["conversation_stage"] = "upselling"
        state["upsell_attempts"] = state.get("upsell_attempts", 0) + 1
        return state
    
    def _finalization_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Finalization Agent node - handles order completion"""
        state["last_agent"] = "finalization"
        state["conversation_stage"] = "finalizing"
        return state
    
    def _delivery_agent_node(self, state: RestaurantState) -> RestaurantState:
        """Delivery Agent node - handles delivery method selection"""
        state["last_agent"] = "delivery"
        state["conversation_stage"] = "awaiting_delivery"
        
        # Check if router detected delivery method
        router_decision = state.get("router_decision", {})
        if router_decision.get("delivery_method"):
            state["delivery_method"] = router_decision["delivery_method"]
        
        return state
    def _greeting_node(self, state: RestaurantState) -> RestaurantState:
        """Handle initial greeting stage"""
        state["conversation_stage"] = "greeting"
        return state
    
    def _menu_browsing_node(self, state: RestaurantState) -> RestaurantState:
        """Handle menu browsing stage"""
        state["conversation_stage"] = "browsing"
        state["menu_displayed"] = True
        return state
    
    def _ordering_node(self, state: RestaurantState) -> RestaurantState:
        """Handle ordering stage"""
        state["conversation_stage"] = "ordering"
        return state
    
    def _upselling_node(self, state: RestaurantState) -> RestaurantState:
        """Handle upselling stage"""
        state["conversation_stage"] = "upselling"
        return state
    
    def _finalizing_node(self, state: RestaurantState) -> RestaurantState:
        """Handle order finalization stage"""
        state["conversation_stage"] = "finalizing"
        return state
    
    def _delivery_method_node(self, state: RestaurantState) -> RestaurantState:
        """Handle delivery method selection stage"""
        state["conversation_stage"] = "awaiting_delivery"
        return state
    
    def _completion_node(self, state: RestaurantState) -> RestaurantState:
        """Handle order completion"""
        state["conversation_stage"] = "completed"
        state["customer_intent"] = "COMPLETED"
        state["last_agent"] = "completion"
        return state
    
    def _human_intervention_node(self, state: RestaurantState) -> RestaurantState:
        """Handle human intervention cases"""
        state["conversation_stage"] = "human_intervention"
        state["customer_intent"] = "HUMAN_NEEDED"
        state["last_agent"] = "human"
        state["needs_intervention"] = True
        return state
    
    def _route_from_router(self, state: RestaurantState) -> str:
        """Route from Router Agent based on its decision"""
        router_decision = state.get("router_decision", {})
        target_agent = router_decision.get("target_agent", "menu")
        
        agent_mapping = {
            "menu": "menu_agent",
            "order": "order_agent",
            "upselling": "upselling_agent",
            "finalization": "finalization_agent",
            "delivery": "delivery_agent",
            "human": "human_intervention"
        }
        
        if state.get("conversation_stage") == "start" or not state.get("conversation_stage"):
            return "greeting"
        
        return agent_mapping.get(target_agent, "menu_agent")
