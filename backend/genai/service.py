"""Post-hoc GenAI explanation & packing-instruction service.

Supports switching dynamically between Google Gemini (`google-genai`) and
Hugging Face serverless small Qwen / fine-tuned models (`InferenceClient`),
strictly after optimization and damage scoring finalize. Falls back cleanly
to an engineering-grade deterministic explanation if credentials or endpoints
are unavailable.
"""

import os
from typing import Optional

from huggingface_hub import InferenceClient

from backend.core.config import SUPPORTED_HF_MODELS, settings

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


class ExplainerService:
    """Generate logistics rationale and operator packing instructions."""

    def __init__(self):
        self.gemini_key = self._resolve_gemini_key()
        self.hf_key = self._resolve_hf_key()

        if self.gemini_key and genai is not None:
            self.gemini_client = genai.Client(api_key=self.gemini_key)
        else:
            self.gemini_client = None

        if self.hf_key:
            self.hf_client = InferenceClient(token=self.hf_key)
        else:
            self.hf_client = None

    @staticmethod
    def _resolve_gemini_key() -> str:
        key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY or ""
        return "" if key in ("", "mock-key-for-mvp", "your-gemini-api-key") else key

    @staticmethod
    def _resolve_hf_key() -> str:
        key = (
            os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HF_TOKEN")
            or settings.HUGGINGFACE_API_KEY
            or ""
        )
        return "" if key in ("", "mock-key-for-mvp", "your-huggingface-token") else key

    def get_provider_status(self) -> dict:
        """Report configured LLM providers and default models."""
        return {
            "gemini_configured": bool(self._resolve_gemini_key() and genai is not None),
            "huggingface_configured": bool(self._resolve_hf_key()),
            "default_gemini_model": settings.GEMINI_MODEL,
            "default_hf_model": settings.HUGGINGFACE_MODEL,
            "supported_hf_models": SUPPORTED_HF_MODELS,
        }

    def generate_summary(
        self,
        carton_id: str,
        risk: float,
        cost: float,
        utilization_pct: float,
        items: Optional[list] = None,
        use_genai: bool = False,
        model_provider: str = "Gemini",
        model_name: Optional[str] = None,
        *,
        box_count: int = 1,
        spec_summary: str = "",
        zone_tier: str = "",
        transit_mode: str = "",
        transit_days: str = "",
        chargeable_wt_kg: float = 0.0,
        volumetric_wt_kg: float = 0.0,
        dunnage_cost: float = 0.0,
        padded_utilization_pct: Optional[float] = None,
    ) -> tuple[str, str]:
        """Return (explanation_markdown, packing_instructions_markdown)."""
        eff_padded = (
            padded_utilization_pct
            if padded_utilization_pct is not None
            else utilization_pct
        )
        container_label = f"{box_count}x {carton_id}" if box_count > 1 else carton_id

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
                    if getattr(i, "requires_cold_chain", False):
                        flags.append("Cold-Chain")
                    qty = getattr(i, "quantity", 1)
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    parts.append(
                        f"{qty}x {i.product_category} (Fragility: {i.fragility_score:.2f}){flag_str}"
                    )
                item_desc = ", ".join(parts)

            prompt = f"""
You are an expert Indian packaging & logistics engineer.
Items packed: {item_desc}.
Selected container: {container_label} ({spec_summary or 'Standard corrugated container'}).
Route & mode: {zone_tier or 'Domestic India'} via {transit_mode or 'Standard'} ({transit_days or 'standard SLA'}).
Weight billing: Chargeable {chargeable_wt_kg:.1f} kg (Volumetric {volumetric_wt_kg:.1f} kg).
Total direct cost: Rs.{cost:,.0f} (includes Rs.{dunnage_cost:,.0f} ISTA void-fill dunnage).
Net volumetric utilization: {utilization_pct*100:.1f}%; Padded occupancy: {eff_padded*100:.1f}%.
Calibrated transit damage risk: {risk*100:.1f}%.

CRITICAL TRANSIT CONTEXT: The package travels across India under +40 C heat,
monsoon humidity (which weakens corrugated ECT by up to 40%), multi-node sorting,
and uneven rural last-mile roads.

Write your response in two sections, separated by the exact delimiter [INSTRUCTIONS].

[ANALYSIS]
Use these EXACT markdown headers:
### Why This Container
### Cost Breakdown
### Safety Assessment
### Environmental Impact

Reference the IS 2771 board spec / cold-chain setup, chargeable vs volumetric weight,
and ISTA 3A cushioning clearances.

At the end of this section, append EXACTLY the following text:

---
### Glossary
- **Dimensional Weight**: Pricing based on outer package volume (`L x W x H / 5000` for parcels or CFT density for PTL) rather than actual weight.
- **Void Fill**: Protective dunnage (EPE foam, air pillows, or kraft paper) filling unprotected space to arrest vibration shift.
- **ESD (Electrostatic Discharge)**: Sudden current that destroys microelectronics; prevented via metallized shielding bags.
- **ECT (Edge Crush Test)**: BIS IS 2771 metric (`kN/m`) measuring corrugated wall stacking strength.

[INSTRUCTIONS]
Write concise, numbered step-by-step warehouse packing instructions addressing ISTA
cushioning thickness, moisture barrier sealing, liquid containment, or thermal packs.
"""

            try:
                content = ""
                provider_label = model_provider or "Gemini"
                is_hf = (
                    provider_label.lower().startswith("hugging")
                    or "qwen" in provider_label.lower()
                )

                if is_hf:
                    hf_key = self._resolve_hf_key()
                    if not hf_key:
                        raise RuntimeError("HUGGINGFACE_API_KEY not configured.")
                    client = (
                        self.hf_client
                        if (self.hf_client is not None and hf_key == self.hf_key)
                        else InferenceClient(token=hf_key)
                    )
                    target_model = (
                        (model_name or "").strip()
                        or settings.HUGGINGFACE_MODEL
                        or "Qwen/Qwen2.5-3B-Instruct"
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert logistics engineer. Follow the exact delimiter and headers.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                    try:
                        response = client.chat_completion(
                            model=target_model,
                            messages=messages,
                            max_tokens=900,
                            temperature=0.3,
                        )
                    except Exception:
                        if target_model != "Qwen/Qwen2.5-7B-Instruct":
                            target_model = "Qwen/Qwen2.5-7B-Instruct"
                            response = client.chat_completion(
                                model=target_model,
                                messages=messages,
                                max_tokens=900,
                                temperature=0.3,
                            )
                        else:
                            raise
                    content = (response.choices[0].message.content or "").strip()
                    header_tag = f"Hugging Face • {target_model}"
                else:
                    gem_key = self._resolve_gemini_key()
                    if not gem_key or genai is None:
                        raise RuntimeError("GEMINI_API_KEY not configured.")
                    client = (
                        self.gemini_client
                        if (self.gemini_client is not None and gem_key == self.gemini_key)
                        else genai.Client(api_key=gem_key)
                    )
                    target_model = (
                        (model_name or "").strip()
                        or settings.GEMINI_MODEL
                        or "gemini-2.5-flash"
                    )
                    response = client.models.generate_content(
                        model=target_model,
                        contents=prompt,
                    )
                    content = (response.text or "").strip()
                    header_tag = f"Gemini • {target_model}"

                if "[INSTRUCTIONS]" in content:
                    parts = content.split("[INSTRUCTIONS]")
                    analysis = (
                        f"**AI Insights ({header_tag}):**\n\n"
                        + parts[0].replace("[ANALYSIS]", "").strip()
                    )
                    instructions = parts[1].strip()
                    return analysis, instructions
                return f"**AI Insights ({header_tag}):**\n\n" + content, ""
            except Exception as exc:
                print(f"GenAI fallback triggered ({model_provider}): {exc}")

        # ------------------------------------------------------------------
        # Engineering-grade deterministic fallback
        # ------------------------------------------------------------------
        void_pct = max(0.0, 100.0 - eff_padded * 100.0)
        spec_note = f" ({spec_summary})" if spec_summary else ""
        route_note = (
            f" on **{zone_tier}** via **{transit_mode}** ({transit_days})"
            if zone_tier and transit_mode
            else ""
        )
        weight_note = (
            f"Chargeable weight is **{chargeable_wt_kg:.1f} kg** "
            f"(volumetric weight **{volumetric_wt_kg:.1f} kg**), "
            if chargeable_wt_kg > 0
            else ""
        )
        dunnage_note = (
            f"including **Rs.{dunnage_cost:,.0f}** in protective void-fill dunnage."
            if dunnage_cost > 0
            else "optimising box-to-payload efficiency."
        )

        fallback_explanation = (
            f"### Why This Container\n"
            f"**{container_label}**{spec_note} achieves **{utilization_pct*100:.1f}% net "
            f"volumetric utilisation** (**{eff_padded*100:.1f}% including ISTA cushioning**), "
            f"leaving {void_pct:.1f}% unprotected void space.\n\n"
            f"### Cost Breakdown\n"
            f"Estimated direct cost: **Rs.{cost:,.0f}**{route_note}. "
            f"{weight_note}{dunnage_note}\n\n"
            f"### Safety Assessment\n"
            f"Calibrated transit damage risk: **{risk*100:.1f}%**. Proper BIS IS 2771 "
            f"board wall strength, monsoon moisture sealing, and ISTA corner clearance "
            f"protect cargo across sorting hubs.\n\n"
            f"### Environmental Impact\n"
            f"Right-sizing to {container_label} minimises wasted dimensional air and "
            f"unnecessary polymer void fill, lowering shipment fuel and material footprint.\n\n"
            f"---\n"
            f"### Glossary\n"
            f"- **Dimensional Weight**: Pricing based on outer package volume "
            f"(`L x W x H / 5000` for parcels or CFT density for PTL) rather than actual weight.\n"
            f"- **Void Fill**: Protective dunnage (EPE foam, air pillows, or kraft paper) "
            f"filling unprotected space to arrest vibration shift.\n"
            f"- **ESD (Electrostatic Discharge)**: Sudden current that destroys "
            f"microelectronics; prevented via metallized shielding bags.\n"
            f"- **ECT (Edge Crush Test)**: BIS IS 2771 metric (`kN/m`) measuring "
            f"corrugated wall stacking strength."
        )
        return fallback_explanation, ""
