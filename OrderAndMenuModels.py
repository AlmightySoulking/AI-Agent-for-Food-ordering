from typing import List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

class MenuCategory(Enum):
    APPETIZERS = "appetizers"
    MAINS = "mains"
    SALADS = "salads"
    DESSERTS = "desserts"
    BEVERAGES = "beverages"
    SIDES = "sides"

class DietaryTag(Enum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    SPICY = "spicy"

@dataclass
class MenuItem:
    id: str
    name: str
    description: str
    price: str
    category: str
    dietary: List[str] = field(default_factory=list)
    popular: bool = False
    chef_recommendation: bool = False
    customizations: List[str] = field(default_factory=list)
    allergens: List[str] = field(default_factory=list)
    prep_time: int = 15  
    spice_level: int = 0

    def __str__(self) -> str:
        base = f"{self.name} -${self.price: 2f}"
        if self.popular:
            base +=  " 🔥"
        if self.chef_recommendation:
            base += " ⭐"
        return base
    
    def get_formatted_description(self) -> str:
        "Get a formatted descriptioon with dietary info"
        desc = self.description
        if self.dietary:
            dietary_tag = ", ".join(self.dietary)
        return desc
    def is_vegetarian(self) -> bool:
        return "vegetarian" in self.dietary
    
    def is_vegan(self) -> bool:
        return "vegan" in self.dietary
    
    def is_gluden_free(self) -> bool:
        return "gluten" not in self.dietary
    
@dataclass
class MenuSection:
    category: MenuCategory
    items: List[MenuItem] = field(default_factory = list)
    description: str = ""

    def add_item(self, item: MenuItem) -> None:
        """Add an Item to this menu section"""
        self.items.append(item)

    def get_popular_items(self) -> List[MenuItem]:
        """Get popular items in this section"""
        return [item for item in self.items if item.popular]
    def get_chef_recommendations(self) -> List[MenuItem]:
        """Get chef's recommended items in this section"""
        return [item for item in self.items if item.chef_recommendation]
    

@dataclass
class OrderItem:
    name: str
    quantity: int
    price: float
    customizations: List[str] = field(default_factory=list)
    special_instructions: str = ""

    def get_total_price(self) -> float:
        return self.price*self.quantity
    
    def __str__(self) -> str:
        base = f"{self.quantity} X {self.name} (${self.price: .2f} each)"
        if self.customizations:
            base +=f" with {', '.join(self.customization)}"
        if self.special_instruction:
            base +=f" - Note: {self.special_instruction}"
        return base
    
@dataclass
class Order:
    items: List[OrderItem] = field(default_factory=list)
    customer_name: str= ""
    order_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending" # pending, confirmed, preparing, ready, delivered
    tax_rate: float = 0.08

    def add_item(self, item: OrderItem) -> None:
        """Add an item to the order"""
        # check if item already exist, if so, update the quantity
        for existing_item in self.items:
            if (existing_item.name == item.name and 
                existing_item.customizations == item.customizations):
                existing_item.quantity += item.quantity
                return
        self.items.append(item)

    def remove_item(self, item_name: str) -> bool:
        """Remove an item from the order"""
        for i, item in enumerate(self.item):
            if item.name.lower() == item_name.lower():
                del self.items[i]
                return True
        return False
    
    def get_subtotal(self) -> float:
        """Get subtotal before tax"""
        return sum(item.get_total_price() for item in self.items)
    
    def get_tax_amount(self) -> float:
        """Calculate tax amount"""
        return self.get_subtotal() * self.tax_rate
    
    def get_total(self) -> float:
        """Get total amount including tax"""
        return self.get_subtotal() + self.get_tax_amount()
    
    def is_empty(self) -> bool:
        """Check if order is empty"""
        return len(self.items) == 0
    
    def clear(self) -> None:
        """Clear all items from order"""
        self.items.clear()

    def __str__(self) -> str:
        if self.is_empty():
            return "Empty order"
        
        order_str = f"Order {self.order_id}:\n"
        for item in self.items:
            order_str += f"  - {item}\n"
        
        order_str += f"\nSubtotal: ${self.get_subtotal():.2f}\n"
        order_str += f"Tax: ${self.get_tax_amount():.2f}\n"
        order_str += f"Total: ${self.get_total():.2f}"
        
        return order_str
