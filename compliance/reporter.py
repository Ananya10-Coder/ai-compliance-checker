import os
import sys
import json
from dotenv import load_dotenv
import re
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from .checker import check_compliance

load_dotenv()
def safe_parse_json(text:str) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM output.
    Attempts direct parsing, then regex extraction of JSON block,
    then falls back to a default structured response.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return {
        "executive_summary": "Error: Could not parse LLM output into valid JSON.",
        "issues": [],
        "raw_llm_output": text
    }

def generate_report(file_path: str):
    """
    Runs compliance check on a file and returns a structured compliance report.
    """
    results = check_compliance(file_path)
    violations_text = ""
    for r in results:
        if r["violations"]:
            violations_text += f"\nChunk ID: {r['chunk_id']}\n"
            violations_text += f"Text: {r['text']}\n"
            violations_text += "Violations:\n"
            for v in r["violations"]:
                violations_text += f"- {v}\n"
    if not violations_text:
        violations_text = "No policy violations were detected in the document."
    
    llm = ChatGroq(
        api_key = os.getenv("GROQ_API_KEY"),
        model = "llama3-8b-8192"
    )

    prompt = f"""
    You are a compliance officer. Analyze the following compliance results.

    For each violation, return structured data in **valid JSON** with this format:

    {{
      "executive_summary": "2–3 sentence summary of findings",
      "issues": [
        {{
          "chunk_id": "<id>",
          "citation": "<exact offending text from document>",
          "violated_policy": "<policy text that was violated>",
          "reasoning": "<why this is a violation>",
          "improvement": "<specific fix>"
        }}
      ]
    }}

    Important:
    - ONLY output valid JSON (no markdown, no commentary).
    - Do not invent unrelated issues. Only use the text provided.
    - One improvement per issue.

    Compliance Results:
    {violations_text}
    """

    response = llm.invoke(prompt)
    parsed_report = safe_parse_json(response.content)
    parsed_report["original_chunks"] = results
    parsed_report["raw_llm_output"] = response.content
    return parsed_report
    

def print_report(report_data: dict):
    """
     Pretty prints the structured compliance report in Markdown style.
    """

    print("\n# Compliance Report\n")
    print("## Executive Summary\n")
    print(report_data.get("executive_summary", "No summary provided."))
    print("\n## Issues & Citations\n")

    for idx, issue in enumerate(report_data.get("issues", []), start=1):
        print(f"### Issue {idx}")
        print(f"- **Chunk**: {issue.get('chunk_id', 'N/A')}")
        print(f"- **Citation**: {issue.get('citation', 'N/A')}")
        print(f"- **Violated Policy**: {issue.get('violated_policy', 'N/A')}")
        print(f"- **Reasoning**: {issue.get('reasoning', 'N/A')}")
        print(f"- **Improvement**: {issue.get('improvement', 'N/A')}")
        print("")