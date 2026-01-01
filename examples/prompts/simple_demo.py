"""
Simple demonstration of prompt generation API.

Shows how to use the prompts module to generate different styles of prompts.
"""

from src.prompts import create_prompt_style, EquationData


def demo_basic_usage():
    """Demonstrate basic prompt generation."""
    print("="*80)
    print("BASIC USAGE DEMO")
    print("="*80)
    
    # Create an equation
    equation = EquationData(
        u="x**2",
        f="x**2 + (x**3)/3",
        kernel="x*t",
        lambda_val=1.0,
        a=0.0,
        b=1.0,
        equation_id="demo_001"
    )
    
    # Generate a prompt
    style = create_prompt_style("chain-of-thought")
    prompt = style.generate(equation, include_ground_truth=True)
    
    print(f"\nEquation ID: {prompt.equation_id}")
    print(f"Style: {prompt.style}")
    print(f"Format: {prompt.format_type}")
    print(f"Ground Truth: {prompt.ground_truth}")
    print(f"\nPrompt:\n{prompt.prompt[:200]}...")


def demo_multiple_styles():
    """Generate prompts in multiple styles."""
    print("\n" + "="*80)
    print("MULTIPLE STYLES DEMO")
    print("="*80)
    
    equation = EquationData(
        u="sin(x)",
        f="sin(x) + cos(x)",
        kernel="cos(x*t)",
        lambda_val=0.5,
        a=0.0,
        b=3.14,
        equation_id="demo_002"
    )
    
    styles = ["basic", "chain-of-thought", "few-shot", "tool-assisted"]
    
    for style_name in styles:
        style = create_prompt_style(style_name)
        prompt = style.generate(equation, include_ground_truth=False)
        print(f"\n[{style_name.upper()}]")
        print(f"Prompt length: {len(prompt.prompt)} characters")


def demo_batch_generation():
    """Generate prompts for multiple equations."""
    print("\n" + "="*80)
    print("BATCH GENERATION DEMO")
    print("="*80)
    
    equations = [
        EquationData(u="x", f="x", kernel="x*t", lambda_val=1.0, a=0, b=1, equation_id=f"batch_{i}")
        for i in range(5)
    ]
    
    style = create_prompt_style("basic")
    prompts = style.generate_batch(equations, include_ground_truth=True)
    
    print(f"\nGenerated {len(prompts)} prompts")
    print(f"All prompts have style: {prompts[0].style}")
    print(f"Equation IDs: {[p.equation_id for p in prompts]}")


def demo_format_types():
    """Generate prompts in different format types."""
    print("\n" + "="*80)
    print("FORMAT TYPES DEMO")
    print("="*80)
    
    equation = EquationData(
        u="exp(x)",
        f="exp(x)",
        kernel="exp(x+t)",
        lambda_val=1.0,
        a=0.0,
        b=1.0,
        equation_id="demo_003"
    )
    
    style = create_prompt_style("basic")
    
    for format_type in ["infix", "latex", "rpn"]:
        prompt = style.generate(equation, format_type=format_type)
        print(f"\n[{format_type.upper()}]")
        print(f"Equation appears as: {format_type}")
        # Show a snippet
        snippet = prompt.prompt[prompt.prompt.find("u(x)"):prompt.prompt.find("u(x)")+100]
        print(f"Snippet: {snippet}...")


def main():
    """Run all demonstrations."""
    demo_basic_usage()
    demo_multiple_styles()
    demo_batch_generation()
    demo_format_types()
    
    print("\n" + "="*80)
    print("For more examples, see examples/prompts/prompt_examples.py")
    print("="*80)


if __name__ == "__main__":
    main()
