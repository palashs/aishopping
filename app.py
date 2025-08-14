import json
import time
import os
import re
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import yaml

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import os
from io import BytesIO
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import Image
from autogen_agentchat.messages import MultiModalMessage  # Import MultiModalMessage
from PIL import Image as PILImage  # Correct import for PIL.Image

from autogen_core import (
    SingleThreadedAgentRuntime,
    TypeSubscription,
    TopicId
)
from autogen_core.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage
)

from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import ChatCompletionClient
from agent_user import UserAgent
from agent_base import AIAgent

from models import UserTask
from topics import (
    triage_agent_topic_type,
    user_topic_type,
    sales_agent_topic_type,
    issues_and_repairs_agent_topic_type,
    product_finder_agent_topic_type,
    product_recommender_agent_topic_type,
    travel_product_recommender_agent_topic_type
)

from tools import (
    execute_order_tool,
    execute_refund_tool,
    look_up_item_tool,
    productfinder_tool
)

from tools_delegate import (
    transfer_to_issues_and_repairs_tool,
    transfer_to_sales_agent_tool,
    transfer_back_to_triage_tool,
    transfer_to_product_finder_agent_tool,
    transfer_to_product_recommender_tool,
    transfer_to_travel_product_recommender_tool
)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import aiofiles
import yaml
import asyncio
from logger import logger_instance
from middleware import LoggingMiddleware

# Logger context for the application
logger_ctx = logger_instance

# Runtime for the agent.
runtime = SingleThreadedAgentRuntime()

# Queue for streaming results from the agent back to the request handler
response_queue: asyncio.Queue[str | object] = asyncio.Queue()

# Sentinel object to signal the end of the stream
STREAM_DONE = object()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        logger_ctx.info("Starting application lifespan")
        # Create chat_history directory if it doesn't exist
        chat_history_dir = "chat_history"
        if not os.path.exists(chat_history_dir):
            os.makedirs(chat_history_dir)

        # Get model client from config.
        async with aiofiles.open("model_config.yaml", "r") as file:
            model_config = yaml.safe_load(await file.read())
        model_client = ChatCompletionClient.load_component(model_config)

        # Register the triage agent.
        triage_agent_type = await AIAgent.register(
            runtime,
            type=triage_agent_topic_type,  # Using the topic type as the agent type.
            factory=lambda: AIAgent(
                description="A triage agent.",
                system_message=SystemMessage(
                    content="You are a customer service bot for CartNexa. "
                    "Introduce yourself. Always be very brief. "
                    "Gather information to direct the customer to the right department. "
                    "But make your questions subtle and natural."
                ),
                model_client=model_client,
                tools=[],
                delegate_tools=[
                    transfer_to_issues_and_repairs_tool,
                    transfer_to_sales_agent_tool,
        ##                transfer_to_product_finder_agent_tool,
                    transfer_to_product_recommender_tool,
        ##                transfer_to_travel_product_recommender_tool
                ],
                agent_topic_type=triage_agent_topic_type,
                user_topic_type=user_topic_type,
                response_queue=response_queue
            ),
        )
        # Add subscriptions for the triage agent: it will receive messages published to its own topic only.
        await runtime.add_subscription(TypeSubscription(topic_type=triage_agent_topic_type, agent_type=triage_agent_type.type))

        # Register the sales agent.
        sales_agent_type = await AIAgent.register(
            runtime,
            type=sales_agent_topic_type,  # Using the topic type as the agent type.
            factory=lambda: AIAgent(
                description="A sales agent.",
                system_message=SystemMessage(
                    content="You are a sales agent for CartNexa Inc."
                    "Always answer in a sentence or less."
                    "Follow the following routine with the user:"
                    "1. Ask them about any problems in their life related to catching roadrunners.\n"
                    "2. Casually mention one of CartNexa's crazy made-up products can help.\n"
                    " - Don't mention price.\n"
                    "3. Once the user is bought in, drop a ridiculous price.\n"
                    "4. Only after everything, and if the user says yes, "
                    "tell them a crazy caveat and execute their order.\n"
                    ""
                ),
                model_client=model_client,
                tools=[execute_order_tool],
                delegate_tools=[transfer_back_to_triage_tool],
                agent_topic_type=sales_agent_topic_type,
                user_topic_type=user_topic_type,
                response_queue=response_queue
            ),
        )
        # Add subscriptions for the sales agent: it will receive messages published to its own topic only.
        await runtime.add_subscription(TypeSubscription(topic_type=sales_agent_topic_type, agent_type=sales_agent_type.type))

        # Register the issues and repairs agent.
        issues_and_repairs_agent_type = await AIAgent.register(
            runtime,
            type=issues_and_repairs_agent_topic_type,  # Using the topic type as the agent type.
            factory=lambda: AIAgent(
                description="An issues and repairs agent.",
                system_message=SystemMessage(
                    content="You are a customer support agent for CartNexa Inc."
                    "Always answer in a sentence or less."
                    "Follow the following routine with the user:"
                    "1. First, ask probing questions and understand the user's problem deeper.\n"
                    " - unless the user has already provided a reason.\n"
                    "2. Propose a fix (make one up).\n"
                    "3. ONLY if not satisfied, offer a refund.\n"
                    "4. If accepted, search for the ID and then execute refund."
                ),
                model_client=model_client,
                tools=[
                    execute_refund_tool,
                    look_up_item_tool,
                ],
                delegate_tools=[transfer_back_to_triage_tool],
                agent_topic_type=issues_and_repairs_agent_topic_type,
                user_topic_type=user_topic_type,
                response_queue=response_queue
            ),
        )
        # Add subscriptions for the issues and repairs agent: it will receive messages published to its own topic only.
        await runtime.add_subscription(
            TypeSubscription(topic_type=issues_and_repairs_agent_topic_type, agent_type=issues_and_repairs_agent_type.type)
        )

        # Register the product finder agent.
        product_finder_agent_type = await AIAgent.register(
            runtime,
            type=product_finder_agent_topic_type,  # Using the topic type as the agent type.
            factory=lambda: AIAgent(
                description="An product finder agent.",
                system_message=SystemMessage(
                    content="You are a helpful AI assistant that filters product categories based on the search string"
                " use functional tool result only and strictly return valid JSON objects."
                ),
                model_client=model_client,
                tools=[
                    productfinder_tool
                ],
                delegate_tools=[transfer_back_to_triage_tool],
                agent_topic_type=product_finder_agent_topic_type,
                user_topic_type=user_topic_type,
                response_queue=response_queue
            ),
        )
        # Add subscriptions for the product finder agent: it will receive messages published to its own topic only.
        await runtime.add_subscription(
            TypeSubscription(topic_type=product_finder_agent_topic_type, agent_type=product_finder_agent_type.type)
        )

        # Register the product recommender agent.
        product_recommender_agent_type = await AIAgent.register(
            runtime,
            type=product_recommender_agent_topic_type,  # Using the topic type as the agent type.
            factory=lambda: AIAgent(
                description="An product recommender agent.",
                system_message=SystemMessage(
                    content="""You are a helpful AI assistant who recommends curated products from the list of products returned by the Product Finder Agent. 
                    DO NOT RETURN ANY PRODUCT FROM OUTSIDE THE LIST. Provide two or more design options. 
                    In each design option, pick maximum one product per category. 
                    Each design option should contain a variety of relevant product categories if available in the list.
                    Return the response in the JSON schema given below. You are part of a bigger system, and you must always respond with a pure JSON response. The system will break if you don't.
                    JSON Schema:
                    {"ReplyMessage":<AgentResponse>,"design_options":[{"design_name":<DesignName>,"design_highlights":<DesignOptionHighlights>,"recommended_products":[{"category":<Category>,"products":[{"name":<ProductName>,"id":<ProductID>,"price":<ProductPrice>,"SuggestedReason":<SuggestedReason>}]}]}]}
                    """
                ),
                model_client=model_client,
                tools=[productfinder_tool],
                delegate_tools=[transfer_back_to_triage_tool],
                agent_topic_type=product_recommender_agent_topic_type,
                user_topic_type=user_topic_type,
                response_queue=response_queue
            ),
        )
        # Add subscriptions for the product finder agent: it will receive messages published to its own topic only.
        await runtime.add_subscription(
            TypeSubscription(topic_type=product_recommender_agent_topic_type, agent_type=product_recommender_agent_type.type)
        )


        # Register the product recommender agent.
        travel_product_recommender_agent_type = await AIAgent.register(
            runtime,
            type=travel_product_recommender_agent_topic_type,  # Using the topic type as the agent type.
            factory=lambda: AIAgent(
                description="An travel product recommender agent.",
                system_message=SystemMessage(
                    content="You are a helpful AI assistant who recommends curated travel related products from the available list of products. Do not recommend any other products which are not in the function tool response list. Return a JSON response in the following format:\n\n"
                    "{\n"
                    "  \"ReplyMessage\": \"<Agent Response>\",\n"
                    "  \"recommended_products\": [\n"
                    "    {\n"
                    "      \"category\": \"<Category>\",\n"
                    "      \"products\": [\n"
                    "        {\n"
                    "          \"name\": \"<Product Name>\",\n"
                    "          \"id\": <Product ID>,\n"
                    "          \"price\": <Product Price>,\n"
                    "          \"SuggestedReason\": \"<Suggested Reason>\"\n"
                    "        }\n"
                    "      ]\n"
                    "    }\n"
                    "  ]\n"
                    "}\n\n"
                    "Respond only with this structured JSON."
                ),
                model_client=model_client,
                tools=[productfinder_tool],
                delegate_tools=[transfer_back_to_triage_tool],
                agent_topic_type=travel_product_recommender_agent_topic_type,
                user_topic_type=user_topic_type,
                response_queue=response_queue
            ),
        )
        # Add subscriptions for the product finder agent: it will receive messages published to its own topic only.
        await runtime.add_subscription(
            TypeSubscription(topic_type=travel_product_recommender_agent_topic_type, agent_type=travel_product_recommender_agent_type.type)
        )

        # Register the user agent.
        user_agent_type = await UserAgent.register(
            runtime,
            type=user_topic_type,
            factory=lambda: UserAgent(
                description="A user agent.",
                user_topic_type=user_topic_type,
                agent_topic_type=triage_agent_topic_type,
                response_queue=response_queue,
                stream_done = STREAM_DONE
            )
        )
        # Add subscriptions for the user agent: it will receive messages published to its own topic only.
        await runtime.add_subscription(TypeSubscription(topic_type=user_topic_type, agent_type=user_agent_type.type))

        # Start the agent runtime.
        runtime.start()
        yield
        await runtime.stop()
        logger_ctx.info("Application lifespan ended")
    except Exception as e:
        logger_ctx.error(f"Error occurred during application lifespan: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


app = FastAPI(lifespan=lifespan)

# Add middleware for logging
app.add_middleware(LoggingMiddleware)


# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

api_router = APIRouter()

@app.get("/index")
async def read_index():
    # Serve the index.html file
    return FileResponse('static/index.html')

@app.get("/hello")
async def hello_world():
    return "Hello, World!"  # This will return a string response


@app.post("/chat/completions")
async def chat_completions_stream(request: Request):
    logger_ctx.info("Starting chat_completions_stream")
    json_data = await request.json()
    message = json_data.get("message", "")
    conversation_id = json_data.get("conversation_id", "conv_id")

    if not isinstance(message, str):
        logger_ctx.error(f"Invalid input: 'message' must be a string.")
        raise HTTPException(status_code=400, detail="Invalid input: 'message' must be a string.")
    
    if not isinstance(conversation_id, str):
        logger_ctx.error(f"Invalid input: 'conversation_id' must be a string.")
        raise HTTPException(status_code=400, detail="Invalid input: 'conversation_id' must be a string.")

    # Validate conversation_id to prevent path traversal attacks
    if not re.match(r'^[A-Za-z0-9_-]+$', conversation_id):
        logger_ctx.error(f"Invalid input: 'conversation_id' contains invalid characters.")
        raise HTTPException(status_code=400, detail="Invalid input: 'conversation_id' contains invalid characters.")

    chat_history_dir = "chat_history"
    base_dir = os.path.abspath(chat_history_dir)
    full_path = os.path.normpath(os.path.join(base_dir, f"history-{conversation_id}.json"))
    if not full_path.startswith(base_dir + os.sep):
        logger_ctx.error(f"Invalid input: 'conversation_id' leads to invalid path.")
        raise HTTPException(status_code=400, detail="Invalid input: 'conversation_id' leads to invalid path.")
    chat_history_file = full_path
    
    messages = []
    # Initialize chat_history and route_agent with default values
    chat_history = {} 
    route_agent = triage_agent_topic_type

    # Load chat history if it exists.
    # Chat history is saved inside the UserAgent. Use redis if possible.
    # There may be a better way to do this.
    if os.path.exists(chat_history_file):
        context = BufferedChatCompletionContext(buffer_size=15)
        try:
            logger_ctx.debug(f"Loading chat history from {chat_history_file}")
            async with aiofiles.open(chat_history_file, "r") as f:
                content = await f.read()
                if content: # Check if file is not empty
                    chat_history = json.loads(content)
                    await context.load_state(chat_history) # Load state only if history is loaded
                    loaded_messages = await context.get_messages()
                    if loaded_messages:
                        messages = loaded_messages
                        last_message = messages[-1]
                        if isinstance(last_message, AssistantMessage) and isinstance(last_message.source, str):
                            route_agent = last_message.source
        except json.JSONDecodeError:
            logger_ctx.error(f"Error decoding JSON from {chat_history_file}. Starting with empty history.")
            # Reset to defaults if loading fails
            messages = []
            route_agent = triage_agent_topic_type
            chat_history = {}
        except Exception as e:
            logger_ctx.error(f"Error loading chat history for {conversation_id}: {e}")
            # Reset to defaults on other errors
            messages = []
            route_agent = triage_agent_topic_type
            chat_history = {}
    # else: route_agent remains the default triage_agent_topic_type if file doesn't exist

    messages.append(UserMessage(content=message,source="User"))

    

    async def response_stream() -> AsyncGenerator[str, None]:
        logger_ctx.info("Starting response stream")
        task1 = asyncio.create_task(runtime.publish_message(
            UserTask(context=messages),
            topic_id=TopicId(type=route_agent, source=conversation_id), # Explicitly use 'type' parameter
        ))
        # Consume items from the response queue until the stream ends or an error occurs
        while True:
            item = await response_queue.get()
            if item is STREAM_DONE:
                logger_ctx.info("Received STREAM_DONE")
                break
            elif isinstance(item, str) and item.startswith("ERROR:"):
                logger_ctx.error(f"Received error message from agent: {item}")
                break
            # Ensure item is serializable before yielding
            else:
                yield json.dumps({"content": item}) + "\n"

        # Wait for the task to finish.
        await task1
        logger_ctx.info("Ending response_stream")


    logger_ctx.info("Ending chat_completions_stream")
    return StreamingResponse(response_stream(), media_type="text/plain")  # type: ignore


# Create directory for uploaded images
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

with open("model_config.yaml", "r") as file:
        model_config = yaml.safe_load(file)
    
        model_client = ChatCompletionClient.load_component(model_config)
# Initialize AutoGen assistant agent for image processing
assistant = AssistantAgent(name="image_description_bot", model_client=model_client)



@app.post("/api/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    logger_ctx.info("Starting image upload")
    if not file:
        logger_ctx.error("File is required")
        raise HTTPException(status_code=400, detail="File is required")
    if not file.content_type.startswith('image/'):
        logger_ctx.error("File type is not an image")
        raise HTTPException(status_code=400, detail="File type is not an image")

    # Save the uploaded file
    logger_ctx.debug(f"Saving uploaded file: {file.filename}")
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as file_object:
        file_object.write(await file.read())

    # Process the uploaded image
    try:
        logger_ctx.info(f"Processing image")
        logger_ctx.debug(f"File location: {file.filename} at {file_location}")
        pil_image = PILImage.open(file_location)
        img = Image(pil_image)

        multi_modal_message = MultiModalMessage(
            content=[
                "Please analyze the image and list out the items in the room. Also, recommend which items should be changed for renovation. Return the response in the following format: "
                '{"info": "File \'' + file.filename + '\' saved at \'' + file_location + '\'",'
                '"itemsInTheRoom":[{"Item":"", "ItemDescription":""}],'
                '"RenovationRecommendations":[{"Item":"", "ProductRecommendations":""}]}',
                img
            ],
            source="user"
        )

        # Get the response from the assistant
        result = await assistant.run(task=multi_modal_message)

        # Ensure the response is not empty
        if not result.messages:
            logger_ctx.error("No response message returned from the assistant.")
            raise ValueError("No response message returned from the assistant.")

        response_content = result.messages[-1].content.strip()
        
        # Remove leading ```json and trailing ```text
        if response_content.startswith("```json"):
            response_content = response_content[7:].strip()  # Remove starting ```json
        if response_content.endswith("```"):
            response_content = response_content[:-3].strip()  # Remove ending ```text

        # Print cleaned response content for debugging
        logger_ctx.debug(f"Cleaned Response Content: {response_content}")

        # Convert the cleaned response into a JSON object
        json_response = json.loads(response_content)

    except json.JSONDecodeError as json_error:
        logger_ctx.error(f"JSON parsing error: {str(json_error)}")
        raise HTTPException(status_code=500, detail=f"JSON parsing error: {str(json_error)}")
    except Exception as e:
        logger_ctx.error(f"Failed to process image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

    logger_ctx.info("End image upload")
    return json_response


# Main route serving the HTML form for image uploads
@app.get("/", response_class=HTMLResponse)
async def main():
    logger_ctx.info("Serving main HTML page for image upload")
    content = """
    <body>
        <h2>Upload an Image</h2>
        <form action="/api/upload-image/" method="post" enctype="multipart/form-data">
            <input name="file" type="file" accept="image/*" required>
            <input type="submit" value="Upload">
        </form>
    </body>
    """
    return content


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)



