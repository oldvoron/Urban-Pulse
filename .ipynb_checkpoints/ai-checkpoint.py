import os
from dotenv import load_dotenv
import anthropic


def get_urban_insights(
    city_name: str,
    poi_stats: dict,
    height_stats: dict,
    network_stats: dict,
) -> str:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key or api_key == "your_key_here":
        return "• Set ANTHROPIC_API_KEY in .env to enable AI insights."

    try:
        client = anthropic.Anthropic(api_key=api_key)

        context = f"""
City: {city_name}

POI distribution (top categories by count):
{poi_stats}

Building heights (metres):
{height_stats}

Street network:
{network_stats}
""".strip()

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system="You are an urban planning analyst. Be sharp, specific, data-driven. No filler.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Based on this spatial data for {city_name}, give exactly 3 bullet-point "
                        "insights about the city's spatial structure. Each bullet must start with '• '.\n\n"
                        + context
                    ),
                }
            ],
        )

        return message.content[0].text

    except Exception as e:
        return f"• AI insights unavailable: {e}"
