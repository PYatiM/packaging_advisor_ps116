import os
# pyrefly: ignore [missing-import]
import google.generativeai as genai
from huggingface_hub import InferenceClient

class ExplainerService:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.hf_key = os.getenv("HUGGINGFACE_API_KEY")

        if self.gemini_key and self.gemini_key != "mock-key-for-mvp":
            genai.configure(api_key=self.gemini_key)
        else:
            self.gemini_key = None

        if self.hf_key and self.hf_key != "mock-key-for-mvp":
            self.hf_client = InferenceClient(token=self.hf_key)
        else:
            self.hf_client = None

    def generate_summary(
        self,
        carton_id: str,
        risk: float,
        cost: float,
        utilization_pct: float,
        items: list = None,
        use_genai: bool = False,
        model_provider: str = "Gemini",
    ) -> tuple[str, str]:
        """Generate an explanation and packing instructions.

        Returns (explanation_markdown, packing_instructions_markdown).
        When GenAI is enabled and a provider is configured, calls the
        LLM.  Otherwise falls back to a deterministic heuristic summary.
        """
        if use_genai:
            item_desc = "Unknown items"
            if items:
                parts = []
                for i in items:
                    flags = []
                    if i.is_liquid:
                        flags.append("Liquid")
                    if i.orientation_sensitive:
                        flags.append("Orientation Sensitive")
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    parts.append(
                        f"{i.product_category} (Fragility: {i.fragility_score:.1f}){flag_str}"
                    )
                item_desc = ", ".join(parts)

            prompt = f"""
You are an expert logistics AI. Items packed: {item_desc}.
Selected container: {carton_id}. Total cost: Rs.{cost:,.0f}. Damage risk: {risk*100:.1f}%.
Volumetric utilisation: {utilization_pct*100:.1f}%.

CRITICAL CONTEXT: The transit environment is INDIA. The package will face
extreme heat (+40 C), heavy monsoon rains/humidity, rough manual handling,
and uneven rural roads.

Write your response in two sections, separated by the exact delimiter
[INSTRUCTIONS].

[ANALYSIS]
Use these EXACT markdown headers:
### Why This Container
### Cost Breakdown
### Safety Assessment
### Environmental Impact

Explain your reasoning clearly and practically based on the specific items.
Mention how the container choice helps survive Indian transit conditions.

At the end of this section, append EXACTLY the following text:

---
### Glossary
- **Dimensional Weight**: Pricing based on package volume rather than
  actual weight. A large, light box costs as much to ship as a small, heavy one.
- **Void Fill**: Material (bubble wrap, paper, foam) used to fill empty
  space and prevent items from shifting.
- **ESD (Electrostatic Discharge)**: A sudden current that can damage
  sensitive electronics. ESD bags block this.
- **Corrugated Cardboard**: Heavy-duty board with internal ridges that
  absorb shocks and resist stacking pressure.

[INSTRUCTIONS]
Write specific, easy-to-follow step-by-step packing instructions.
Address any fragile, liquid, or orientation needs clearly.
"""

            try:
                content = ""
                if model_provider == "Hugging Face" and self.hf_client:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert logistics AI. "
                                "Follow the user's formatting instructions exactly."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                    response = self.hf_client.chat_completion(
                        model="PYatiM/packaging_advisor_ps116",
                        messages=messages,
                        max_tokens=1000,
                    )
                    content = response.choices[0].message.content.strip()
                elif self.gemini_key:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    content = response.text.strip()
                else:
                    raise Exception("No valid AI provider configured.")

                if "[INSTRUCTIONS]" in content:
                    parts = content.split("[INSTRUCTIONS]")
                    analysis = (
                        f"**AI Insights ({model_provider}):**\n\n"
                        + parts[0].replace("[ANALYSIS]", "").strip()
                    )
                    instructions = parts[1].strip()
                    return analysis, instructions
                else:
                    return (
                        f"**AI Insights ({model_provider}):**\n\n" + content,
                        "",
                    )
            except Exception as e:
                print(f"GenAI failed: {e}")
                # fall through to heuristic below

        # --------------------------------------------------------------
        # Deterministic fallback
        # --------------------------------------------------------------
        void_pct = 100.0 - utilization_pct * 100
        fallback_explanation = (
            f"### Why This Container\n"
            f"**{carton_id}** achieves **{utilization_pct*100:.1f}% volumetric "
            f"utilisation**, leaving {void_pct:.1f}% space for protective "
            f"void-fill without excessive empty gaps where items could shift.\n\n"
            f"### Cost Breakdown\n"
            f"Estimated cost: **Rs.{cost:,.0f}**. By selecting the tightest "
            f"fitting container we minimise dimensional-weight surcharges — "
            f"you are not paying to ship empty air.\n\n"
            f"### Safety Assessment\n"
            f"Calculated damage risk: **{risk*100:.1f}%** based on {void_pct:.1f}% "
            f"void space and item fragility. Following the packing instructions "
            f"below will further reduce this risk.\n\n"
            f"### Environmental Impact\n"
            f"A well-fitted container at {utilization_pct*100:.1f}% efficiency "
            f"reduces the need for excess packing material, lowering waste and "
            f"overall shipment carbon footprint.\n\n"
            f"---\n"
            f"### Glossary\n"
            f"- **Dimensional Weight**: Pricing based on package volume rather than "
            f"actual weight. A large, light box costs as much to ship as a small, heavy one.\n"
            f"- **Void Fill**: Material (bubble wrap, paper, foam) used to fill "
            f"empty space and prevent items from shifting.\n"
            f"- **ESD (Electrostatic Discharge)**: A sudden current that can "
            f"damage sensitive electronics. ESD bags block this.\n"
            f"- **Corrugated Cardboard**: Heavy-duty board with internal ridges "
            f"that absorb shocks and resist stacking pressure."
        )
        return fallback_explanation, ""
