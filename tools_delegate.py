from autogen_core.tools import FunctionTool
from topics import product_finder_agent_topic_type, product_recommender_agent_topic_type, sales_agent_topic_type, issues_and_repairs_agent_topic_type,triage_agent_topic_type,travel_product_recommender_agent_topic_type

def transfer_to_sales_agent() -> str:
    return sales_agent_topic_type

def transfer_to_issues_and_repairs() -> str:
    return issues_and_repairs_agent_topic_type


def transfer_to_product_finder_agent() -> str:
    return product_finder_agent_topic_type


def transfer_to_product_recommender() -> str:
    return product_recommender_agent_topic_type

def transfer_to_travel_product_recommender() -> str:
    return travel_product_recommender_agent_topic_type

def transfer_back_to_triage() -> str:
    return triage_agent_topic_type


transfer_to_sales_agent_tool = FunctionTool(
    transfer_to_sales_agent, description="Use for anything sales or buying related."
)
transfer_to_issues_and_repairs_tool = FunctionTool(
    transfer_to_issues_and_repairs, description="Use for issues, repairs, or refunds."
)

transfer_to_product_finder_agent_tool = FunctionTool(
    transfer_to_product_finder_agent, description="Use for anything products finding related."
)
transfer_to_product_recommender_tool = FunctionTool(
    transfer_to_product_recommender, description="Use for products recommendations related."
)
transfer_to_travel_product_recommender_tool = FunctionTool(
    transfer_to_travel_product_recommender, description="Use for travel required products recommendations related."
)
transfer_back_to_triage_tool = FunctionTool(
    transfer_back_to_triage,
    description="Call this if the user brings up a topic outside of your purview,\nincluding escalating to human.",
)
