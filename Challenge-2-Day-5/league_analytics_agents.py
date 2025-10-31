import os
from strands import Agent, tool
from strands.models import BedrockModel
from strands.memory import ConversationMemory
import boto3
from typing import Dict, Any

# Configuration - Replace with your actual values
KNOWLEDGE_BASE_ID = "EA5B00GIAI"  # From Day 4 setup
REGION = "ap-south-1"  # Your AWS region
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"  # or whatever model you like

# Initialize Bedrock model
model = BedrockModel(
    model_id=MODEL_ID,
    region_name=REGION,
)

# Create the basic agent
agent = Agent(
    model=model,
    system_prompt="""You are a League of Legends analytics expert. You have access to match data 
    through a knowledge base and can perform sophisticated analysis of champion performance, 
    meta trends, and gameplay patterns. Always provide specific data-driven insights and 
    cite your sources when making claims about game statistics.""",
)


# Initialize Bedrock client
bedrock_client = boto3.client("bedrock-agent-runtime", region_name=REGION)


@tool
def query_match_data(query: str, max_results: int = 5) -> str:
    """
    Query the League of Legends match data knowledge base for specific information.

    Args:
        query: The question about match data, champion performance, or meta analysis
        max_results: Number of relevant documents to retrieve (default: 5)

    Returns:
        Detailed analysis based on the match data
    """
    try:
        response = bedrock_client.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ID,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {"numberOfResults": max_results}
                    },
                },
            },
        )

        # Extract the generated answer
        answer = response.get("output", {}).get("text", "")

        # Get source citations for transparency
        citations = []
        for citation in response.get("citations", []):
            for ref in citation.get("retrievedReferences", []):
                citations.append(
                    {
                        "content_snippet": ref.get("content", {}).get("text", "")[:200]
                        + "...",
                        "metadata": ref.get("metadata", {}),
                    }
                )

        result = f"Analysis: {answer}\n\nSources: {len(citations)} match data references used"
        return result

    except Exception as e:
        return f"Error querying match data: {str(e)}"


# Add the tool to the agent
agent.tools.append(query_match_data)


@tool
def analyze_champion_performance(
    champion_name: str, role: str = None, rank_tier: str = None
) -> str:
    """
    Analyze detailed performance metrics for a specific champion.

    Args:
        champion_name: Name of the champion to analyze
        role: Specific role/position (optional)
        rank_tier: Rank tier to filter by (optional)

    Returns:
        Comprehensive champion performance analysis
    """
    query_parts = [f"champion performance analysis for {champion_name}"]

    if role:
        query_parts.append(f"in the {role} role")
    if rank_tier:
        query_parts.append(f"in {rank_tier} ranked games")

    query_parts.extend(
        [
            "including win rate, KDA ratios, item build success rates,",
            "pick rate trends, ban rate, and comparison to other champions in the same role",
        ]
    )

    full_query = " ".join(query_parts)
    return query_match_data(full_query, max_results=8)


@tool
def analyze_meta_trends(
    patch_version: str = None, time_period: str = "last 30 days"
) -> str:
    """
    Analyze current meta trends and shifts in champion popularity.

    Args:
        patch_version: Specific patch to analyze (optional)
        time_period: Time period for trend analysis

    Returns:
        Meta trend analysis with rising/falling champions
    """
    query_parts = ["meta trends analysis showing"]

    if patch_version:
        query_parts.append(f"changes in patch {patch_version}")
    else:
        query_parts.append(f"trends over the {time_period}")

    query_parts.extend(
        [
            "including champion pick rate changes, win rate shifts,",
            "emerging strategies, item build evolution, and role meta shifts",
        ]
    )

    full_query = " ".join(query_parts)
    return query_match_data(full_query, max_results=10)


@tool
def compare_team_compositions(comp1_description: str, comp2_description: str) -> str:
    """
    Compare the effectiveness of different team compositions.

    Args:
        comp1_description: Description of first team composition
        comp2_description: Description of second team composition

    Returns:
        Comparative analysis of team composition effectiveness
    """
    query = f"""
    Compare team composition effectiveness between:
    Composition 1: {comp1_description}
    Composition 2: {comp2_description}
    
    Include win rates, synergy analysis, power spikes, team fight effectiveness,
    objective control, and matchup considerations
    """

    return query_match_data(query, max_results=12)


# Add specialized tools to agent
agent.tools.extend(
    [analyze_champion_performance, analyze_meta_trends, compare_team_compositions]
)

# Configure conversation memory for context retention
memory = ConversationMemory(
    memory_size=10, include_system_messages=True  # Keep last 10 exchanges
)

# Update agent with memory capabilities
agent = Agent(
    model=model,
    memory=memory,
    system_prompt="""You are a League of Legends analytics expert with conversation memory. 
    Remember previous questions and analyses within our conversation to provide contextual responses.
    Reference past discussions when relevant and build upon previous insights.""",
    tools=[
        query_match_data,
        analyze_champion_performance,
        analyze_meta_trends,
        compare_team_compositions,
    ],
)


# Simple context-aware message handler
@agent.on_message
async def handle_message_with_memory(message: str):
    """Process messages with conversation context"""

    # Get recent conversation for context
    recent_messages = agent.memory.get_recent_messages(limit=3)

    # Check for follow-up patterns
    if any(
        word in message.lower()
        for word in ["that champion", "this build", "same", "also"]
    ):
        message += " (Reference previous discussion context)"

    print(f"Processing query with memory context: {message[:100]}...")
    return await agent.process_message(message)


# Add conversation handlers
@agent.on_message
async def handle_message(message: str):
    """Pre-process user input for better analysis"""
    # Log the interaction
    print(f"Processing query: {message[:100]}...")

    # Enhance queries with context
    if "champion" in message.lower() and "vs" in message.lower():
        message += " Include matchup statistics and win rate comparisons."
    elif "meta" in message.lower():
        message += " Focus on recent trends and statistical significance."
    return await agent.process_message(message)


@agent.on_error
async def handle_error(error: Exception):
    """Handle errors gracefully with helpful suggestions"""
    error_msg = str(error)
    if "knowledge base" in error_msg.lower():
        return {
            "error": "Unable to access match data",
            "suggestion": "Please verify your Knowledge Base is active and try a more specific query",
        }
    elif "timeout" in error_msg.lower():
        return {
            "error": "Query took too long to process",
            "suggestion": "Try breaking your question into smaller, more focused queries",
        }
    else:
        return {
            "error": f"Analysis error: {error_msg}",
            "suggestion": "Please rephrase your question or try a different approach",
        }


async def test_agent():
    """Test the agent with various League of Legends queries"""

    test_queries = [
        "What are the most successful item builds for Jinx in ranked games?",
        "Show me the current meta trends for ADC champions",
        "Compare the effectiveness of engage supports versus enchanter supports",
        "Which champions have the highest win rate in Diamond+ games?",
        "Analyze the performance of Azir in mid lane during the current patch",
    ]

    print("🧪 Testing League Analytics Agent...")

    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\n--- Test {i}: {query} ---")
            response = await agent.process_message(query)
            print(f"✅ Response: {response[:200]}...")
            metrics.log_query(True)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            metrics.log_query(False)
    print(
        f"\n📊 Test Summary: {metrics.successful_queries}/{metrics.query_count} successful"
    )


# Run tests
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
