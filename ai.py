import os
from dotenv import load_dotenv
import anthropic


def get_urban_insights(
    city_name: str,
    poi_stats: dict,
    height_stats: dict,
    network_stats: dict,
    morph_stats: dict = None,
) -> str:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key or api_key == "your_key_here":
        return "• Set ANTHROPIC_API_KEY in .env to enable AI insights."

    try:
        client = anthropic.Anthropic(api_key=api_key)

        morph_block = ""
        if morph_stats:
            morph_block = f"\nMorphology & land use:\n{morph_stats}"

        context = f"""
City: {city_name}

POI distribution (top categories by count):
{poi_stats}

Building heights (metres):
{height_stats}

Street network:
{network_stats}{morph_block}
""".strip()

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system="You are an urban planning analyst. Be sharp, specific, data-driven. No filler.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Based on this spatial data for {city_name}, give exactly 5 bullet-point "
                        "insights about the city's spatial structure, morphology, and urban form. "
                        "Each bullet must start with '• '. Cover: density patterns, street network "
                        "character, green space quality, dominant building typology, and one "
                        "planning recommendation.\n\n" + context
                    ),
                }
            ],
        )

        return message.content[0].text

    except Exception as e:
        return f"• AI insights unavailable: {e}"
